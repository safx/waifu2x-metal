//
//  main.swift
//  waifu2x-metal
//
//  Created by Safx Developer on 2015/06/20.
//  Copyright © 2015年 Safx Developers. All rights reserved.
//


import Cocoa
import Quartz
import Metal
import MetalKit
import simd



extension MTLComputeCommandEncoder {
    func setThread(_ texture: MTLTexture) {
        let width  = texture.width
        let height = texture.height

        let threadsPerThreadgroup = MTLSize(width: 32, height: 16, depth: 1)
        let numGroups = MTLSize(width: 1 + width / threadsPerThreadgroup.width,
            height: 1 + height / threadsPerThreadgroup.height, depth: 1)
        dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadsPerThreadgroup)
    }
}


let device = MTLCreateSystemDefaultDevice()!
let library = device.newDefaultLibrary()!
let queue = device.makeCommandQueue()

let waifu2xPipelineState = try! device.makeComputePipelineState(function: library.makeFunction(name: "waifu2x")!)
let splitToRGBChannelsPipelineState = try! device.makeComputePipelineState(function: library.makeFunction(name: "splitToRGBChannels")!)
let combineRGBChannelsPipelineState = try! device.makeComputePipelineState(function: library.makeFunction(name: "combineRGBChannels")!)


func saveImage(_ image: CGImage, path: String) {
    let rep = NSBitmapImageRep(cgImage: image)
    rep.size = CGSize(width: image.width, height: image.height)

    guard let data = rep.representation(using: .PNG, properties: [:]) else {
        fatalError()
    }
    try? data.write(to: URL(fileURLWithPath: path), options: [.atomic])
}

func createContext(_ texture: MTLTexture) -> CGContext? {
    let width = texture.width
    let height = texture.height
    let rowBytes = width * 4

    var buf = Array<UInt8>(repeating: 0, count: rowBytes * height * 4)
    let region = MTLRegionMake2D(0, 0, width, height)
    texture.getBytes(&buf, bytesPerRow: rowBytes, from: region, mipmapLevel: 0)

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    return CGContext(data: &buf, width: width, height: height, bitsPerComponent: 8, bytesPerRow: rowBytes, space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
}

func createImage(_ texture: MTLTexture) -> CGImage? {
    let context = createContext(texture)
    return context?.makeImage()
}

func createTexture(_ image: CGImage, width: Int, height: Int) -> MTLTexture {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitsPerComp = 8
    let rowBytes = width * 4
    let alpha = CGImageAlphaInfo.premultipliedLast
    let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: bitsPerComp, bytesPerRow: rowBytes, space: colorSpace, bitmapInfo: alpha.rawValue)

    context!.interpolationQuality = .none
    context?.draw(image, in: CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))

    let texture = createEmptyTexture(device, width: width, height: height)
    let pixels = context?.data
    let region = MTLRegionMake2D(0, 0, width, height)
    texture.replace(region: region, mipmapLevel: 0, withBytes: pixels!, bytesPerRow: rowBytes)

    return texture
}

func loadImage(_ path: String) -> CGImage? {
    let data = try? Data(contentsOf: URL(fileURLWithPath: path))
    guard let imgDataProvider = CGDataProvider(data: data as! CFData) else {
        fatalError("ImageProvider failure")
    }
    if let image = CGImage(jpegDataProviderSource: imgDataProvider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) { return image }
    let image = CGImage(pngDataProviderSource: imgDataProvider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)

    return image
}

func createEmptyTexture(_ device: MTLDevice, width: Int, height: Int, format: MTLPixelFormat = .rgba8Unorm, length: Int = 0) -> MTLTexture {
    let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: false)
    if length > 0 {
        desc.textureType = .type2DArray
        desc.arrayLength = length
    }

    return device.makeTexture(descriptor: desc)
}

func createEncoder(_ commandBuffer: MTLCommandBuffer, pipelineState: MTLComputePipelineState) -> MTLComputeCommandEncoder {
    let encoder = commandBuffer.makeComputeCommandEncoder()
    encoder.setComputePipelineState(pipelineState)
    return encoder
}


func createMetalFunc(_ pipelineState: MTLComputePipelineState, inTexture: MTLTexture, outTexture: MTLTexture) -> MTLTexture {
    let commandBuf = queue.makeCommandBuffer()
    do {
        let encoder = createEncoder(commandBuf, pipelineState: pipelineState)
        encoder.setTexture(inTexture, at: 0)
        encoder.setTexture(outTexture, at: 1)
        encoder.setThread(inTexture)
        encoder.endEncoding()
    }
    commandBuf.commit()
    commandBuf.waitUntilCompleted()

    if let err = commandBuf.error {
        fatalError("MetalExecutionError: \(err)")
    }

    return outTexture
}

func splitToRGBChannels(_ inTexture: MTLTexture) -> MTLTexture {
    let outTexture = createEmptyTexture(device, width: inTexture.width, height: inTexture.height, format: .r32Float, length: 3)
    return createMetalFunc(splitToRGBChannelsPipelineState, inTexture: inTexture, outTexture: outTexture)
}

func combineRGBChannels(_ inTexture: MTLTexture) -> MTLTexture {
    let outTexture = createEmptyTexture(device, width: inTexture.width, height: inTexture.height)
    return createMetalFunc(combineRGBChannelsPipelineState, inTexture: inTexture, outTexture: outTexture)
}

func sync(_ texture: MTLTexture) -> MTLTexture {
    let commandBuf = queue.makeCommandBuffer()
    do {
        let encoder = commandBuf.makeBlitCommandEncoder()
        encoder.synchronize(resource: texture)
        encoder.endEncoding()
    }
    commandBuf.commit()
    commandBuf.waitUntilCompleted()

    if let err = commandBuf.error {
        fatalError("MetalExecutionError: \(err)")
    }
    return texture
}

func waifu2x(_ inTextures: MTLTexture, weight: ContiguousArray<float3x3>, bias: Float) -> MTLTexture {
    let inCount = weight.count
    precondition(inCount == inTextures.arrayLength)

    let outTexture = createEmptyTexture(device, width: inTextures.width, height: inTextures.height, format: .r32Float)
    let commandBuf = queue.makeCommandBuffer()

    do {
        let encoder = createEncoder(commandBuf, pipelineState: waifu2xPipelineState)
        encoder.setTexture(inTextures, at: 0)
        encoder.setTexture(outTexture, at: 1)

        let weightBuffer = device.makeBuffer(length: MemoryLayout<float3x3>.size * inCount, options: [])

        let ws = weightBuffer.contents().assumingMemoryBound(to: float3x3.self)
        for i in 0..<inCount { ws[i] = weight[i] }
        encoder.setBuffer(weightBuffer, offset: 0, at: 0)

        let biasBuffer = device.makeBuffer(length: MemoryLayout<Float32>.size, options: [])

        let b = biasBuffer.contents().assumingMemoryBound(to: Float32.self)
        b[0] = Float32(bias)
        encoder.setBuffer(biasBuffer, offset: 0, at: 1)

        encoder.setThread(inTextures)
        encoder.endEncoding()
    }
    commandBuf.commit()
    commandBuf.waitUntilCompleted()

    if let err = commandBuf.error {
        fatalError("MetalExecutionError: \(err)")
    }
    
    return outTexture
}

struct ModelLayer {
    let bias: Array<Float>
    let weight: ContiguousArray<ContiguousArray<float3x3>>
}

func process(_ layerInfo: [ModelLayer], inTexture: MTLTexture) -> MTLTexture {
    var inputs = inTexture

    for info in layerInfo {
        let outputs = zip(info.weight, info.bias).map { (a,b) -> MTLTexture in
            return waifu2x(inputs, weight: a, bias: b)
        }
        print(outputs.count)

        inputs = newArray(outputs[0], size: outputs.count)
        copy(outputs, dest: inputs)
    }

    precondition(inputs.arrayLength == 3)
    return inputs
}


func createModelLayer(_ layerInfo: [String: AnyObject]) -> ModelLayer {
    let kW = layerInfo["kW"] as! Int
    let kH = layerInfo["kH"] as! Int
    precondition(kW == 3 && kH == 3)

    let bias = layerInfo["bias"] as! Array<Float>
    let nInputPlane = layerInfo["nInputPlane"] as! Int
    let nOutputPlane = layerInfo["nOutputPlane"] as! Int
    let weightOrig = layerInfo["weight"] as! [[[[Float]]]]
    var weight = ContiguousArray<ContiguousArray<float3x3>>(repeating: ContiguousArray<float3x3>(), count: nOutputPlane)

    precondition(weightOrig.count == nOutputPlane)
    precondition(weightOrig[0].count == nInputPlane)

    for i in 0..<nOutputPlane {
        weight[i] = ContiguousArray<float3x3>(repeating: float3x3(), count: nInputPlane)
        for j in 0..<nInputPlane {
            let w = weightOrig[i][j]
            weight[i][j] = float3x3([float3(w[0]), float3(w[1]), float3(w[2])])
        }
    }

    return ModelLayer(bias: bias, weight: weight)
}

func newArray(_ e: MTLTexture, size: Int = 1) -> MTLTexture {
    let width = e.width
    let height = e.height
    return createEmptyTexture(device, width: width, height: height, format: .r32Float, length: size)
}

func copy(_ src: [MTLTexture], dest: MTLTexture) {
    let commandBuf = queue.makeCommandBuffer()
    let encoder = commandBuf.makeBlitCommandEncoder()
    for i in 0..<src.count {
        encoder.copy(from: src[i], sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(), sourceSize: MTLSizeMake(dest.width, dest.height, 1), to: dest, destinationSlice: i, destinationLevel: 0, destinationOrigin: MTLOrigin())
    }
    encoder.endEncoding()
    commandBuf.commit()
    commandBuf.waitUntilCompleted()

    if let err = commandBuf.error {
        fatalError("MetalExecutionError: \(err)")
    }
}

if CommandLine.arguments.count >= 1 {
    let path = "a.png"//Process.arguments[1]

    let resizeJsonFile = "scale2.0x_model.json"
    let jsonObj = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: resizeJsonFile)), options: []) as! [[String: AnyObject]]
    let model = jsonObj.map(createModelLayer)

    let image = loadImage(path)!
    let width = image.width
    let height = image.height
    let resized = createTexture(image, width: width * 2, height: height * 2)

    let rgbtex = splitToRGBChannels(resized)
    let tex = process(model, inTexture: rgbtex)
    let outTexture = combineRGBChannels(tex)

    if let image = createImage(sync(outTexture)) {
        saveImage(image, path: "out.png")
    } else {
        print("Image creation failed")
    }
}
