//
//  main.swift
//  waifu2x-metal
//
//  Created by Safx Developer on 2015/06/20.
//  Copyright © 2015年 Safx Developers. All rights reserved.
//


import Cocoa
import Metal
import MetalKit
import simd



extension MTLComputeCommandEncoder {
    func setThread(texture: MTLTexture) {
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
let queue = device.newCommandQueue()

let waifu2xPipelineState = try! device.newComputePipelineStateWithFunction(library.newFunctionWithName("waifu2x")!)
let splitToRGBChannelsPipelineState = try! device.newComputePipelineStateWithFunction(library.newFunctionWithName("splitToRGBChannels")!)
let combineRGBChannelsPipelineState = try! device.newComputePipelineStateWithFunction(library.newFunctionWithName("combineRGBChannels")!)


func saveImage(image: CGImage, path: String) {
    let rep = NSBitmapImageRep(CGImage: image)
    rep.size = CGSize(width: CGImageGetWidth(image), height: CGImageGetHeight(image))

    guard let data = rep.representationUsingType(.NSPNGFileType, properties: [:]) else {
        fatalError()
    }
    data.writeToFile(path, atomically: true)
}

func createContext(texture: MTLTexture) -> CGContext? {
    let width = texture.width
    let height = texture.height
    let rowBytes = width * 4

    var buf = Array<UInt8>(count: rowBytes * height * 4, repeatedValue: 0)
    let region = MTLRegionMake2D(0, 0, width, height)
    texture.getBytes(&buf, bytesPerRow: rowBytes, fromRegion: region, mipmapLevel: 0)

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    return CGBitmapContextCreate(&buf, width, height, 8, rowBytes, colorSpace, CGImageAlphaInfo.PremultipliedLast.rawValue)
}

func createImage(texture: MTLTexture) -> CGImage? {
    let context = createContext(texture)
    return CGBitmapContextCreateImage(context)
}

func createTexture(image: CGImage, width: Int, height: Int) -> MTLTexture {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitsPerComp = 8
    let rowBytes = width * 4
    let alpha = CGImageAlphaInfo.PremultipliedLast
    let context = CGBitmapContextCreate(nil, width, height, bitsPerComp, rowBytes, colorSpace, alpha.rawValue)

    CGContextSetInterpolationQuality(context, .None)
    CGContextDrawImage(context, CGRectMake(0, 0, CGFloat(width), CGFloat(height)), image)

    let texture = createEmptyTexture(device, width: width, height: height)
    let pixels = CGBitmapContextGetData(context)
    let region = MTLRegionMake2D(0, 0, width, height)
    texture.replaceRegion(region, mipmapLevel: 0, withBytes: pixels, bytesPerRow: rowBytes)

    return texture
}

func getPixelBuffer(context: CGContextRef) -> UnsafeMutableBufferPointer<UInt8> {
    let rowBytes = CGBitmapContextGetBytesPerRow(context)
    let height = CGBitmapContextGetHeight(context)
    let data = CGBitmapContextGetData(context)
    let buffer = UnsafeMutableBufferPointer(start: data, count: rowBytes * height)
    return unsafeBitCast(buffer, UnsafeMutableBufferPointer<UInt8>.self)
}

func loadImage(path: String) -> CGImage? {
    let imgDataProvider = CGDataProviderCreateWithCFData(NSData(contentsOfFile: path))
    if let image = CGImageCreateWithJPEGDataProvider(imgDataProvider, nil, true, .RenderingIntentDefault) { return image }
    let image = CGImageCreateWithPNGDataProvider(imgDataProvider, nil, true, .RenderingIntentDefault)

    return image
}

func createEmptyTexture(device: MTLDevice, width: Int, height: Int, format: MTLPixelFormat = .RGBA8Unorm, length: Int = 0) -> MTLTexture {
    let desc = MTLTextureDescriptor.texture2DDescriptorWithPixelFormat(format, width: width, height: height, mipmapped: false)
    if length > 0 {
        desc.textureType = .Type2DArray
        desc.arrayLength = length
    }

    return device.newTextureWithDescriptor(desc)
}

func createEncoder(commandBuffer: MTLCommandBuffer, pipelineState: MTLComputePipelineState) -> MTLComputeCommandEncoder {
    let encoder = commandBuffer.computeCommandEncoder()
    encoder.setComputePipelineState(pipelineState)
    return encoder
}


func createMetalFunc(pipelineState: MTLComputePipelineState, inTexture: MTLTexture, outTexture: MTLTexture) -> MTLTexture {
    let commandBuf = queue.commandBuffer()
    do {
        let encoder = createEncoder(commandBuf, pipelineState: pipelineState)
        encoder.setTexture(inTexture, atIndex: 0)
        encoder.setTexture(outTexture, atIndex: 1)
        encoder.setThread(inTexture)
        encoder.endEncoding()
    }
    commandBuf.commit()
    commandBuf.waitUntilCompleted()

    if let err = commandBuf.error {
        fatalError("MetalExecutionError: " + err.description)
    }

    return outTexture
}

func splitToRGBChannels(inTexture: MTLTexture) -> MTLTexture {
    let outTexture = createEmptyTexture(device, width: inTexture.width, height: inTexture.height, format: .R32Float, length: 3)
    return createMetalFunc(splitToRGBChannelsPipelineState, inTexture: inTexture, outTexture: outTexture)
}

func combineRGBChannels(inTexture: MTLTexture) -> MTLTexture {
    let outTexture = createEmptyTexture(device, width: inTexture.width, height: inTexture.height)
    return createMetalFunc(combineRGBChannelsPipelineState, inTexture: inTexture, outTexture: outTexture)
}

func sync(texture: MTLTexture) -> MTLTexture {
    let commandBuf = queue.commandBuffer()
    do {
        let encoder = commandBuf.blitCommandEncoder()
        encoder.synchronizeResource(texture)
        encoder.endEncoding()
    }
    commandBuf.commit()
    commandBuf.waitUntilCompleted()

    if let err = commandBuf.error {
        fatalError("MetalExecutionError: " + err.description)
    }
    return texture
}

func waifu2x(inTextures: MTLTexture, weight: ContiguousArray<float3x3>, bias: Float) -> MTLTexture {
    let inCount = weight.count
    precondition(inCount == inTextures.arrayLength)

    let outTexture = createEmptyTexture(device, width: inTextures.width, height: inTextures.height, format: .R32Float)
    let commandBuf = queue.commandBuffer()

    do {
        let encoder = createEncoder(commandBuf, pipelineState: waifu2xPipelineState)
        encoder.setTexture(inTextures, atIndex: 0)
        encoder.setTexture(outTexture, atIndex: 1)

        let weightBuffer = device.newBufferWithLength(sizeof(float3x3) * inCount, options: [])
        let ws = UnsafeMutablePointer<float3x3>(weightBuffer.contents())
        for i in 0..<inCount { ws[i] = weight[i] }
        encoder.setBuffer(weightBuffer, offset: 0, atIndex: 0)

        let biasBuffer = device.newBufferWithLength(sizeof(Float32), options: [])
        let b = UnsafeMutablePointer<Float32>(biasBuffer.contents())
        b[0] = Float32(bias)
        encoder.setBuffer(biasBuffer, offset: 0, atIndex: 1)

        encoder.setThread(inTextures)
        encoder.endEncoding()
    }
    commandBuf.commit()
    commandBuf.waitUntilCompleted()

    if let err = commandBuf.error {
        fatalError("MetalExecutionError: " + err.description)
    }
    
    return outTexture
}

struct ModelLayer {
    let bias: Array<Float>
    let weight: ContiguousArray<ContiguousArray<float3x3>>
}

func process(layerInfo: [ModelLayer], inTexture: MTLTexture) -> MTLTexture {
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


func createModelLayer(layerInfo: [String: AnyObject]) -> ModelLayer {
    let kW = layerInfo["kW"] as! Int
    let kH = layerInfo["kH"] as! Int
    precondition(kW == 3 && kH == 3)

    let bias = layerInfo["bias"] as! Array<Float>
    let nInputPlane = layerInfo["nInputPlane"] as! Int
    let nOutputPlane = layerInfo["nOutputPlane"] as! Int
    let weightOrig = layerInfo["weight"] as! [[[[Float]]]]
    var weight = ContiguousArray<ContiguousArray<float3x3>>(count: nOutputPlane, repeatedValue: ContiguousArray<float3x3>())

    precondition(weightOrig.count == nOutputPlane)
    precondition(weightOrig[0].count == nInputPlane)

    for i in 0..<nOutputPlane {
        weight[i] = ContiguousArray<float3x3>(count: nInputPlane, repeatedValue: float3x3())
        for j in 0..<nInputPlane {
            let w = weightOrig[i][j]
            weight[i][j] = float3x3([float3(w[0]), float3(w[1]), float3(w[2])])
        }
    }

    return ModelLayer(bias: bias, weight: weight)
}

func newArray(e: MTLTexture, size: Int = 1) -> MTLTexture {
    let width = e.width
    let height = e.height
    return createEmptyTexture(device, width: width, height: height, format: .R32Float, length: size)
}

func copy(src: [MTLTexture], dest: MTLTexture) {
    let commandBuf = queue.commandBuffer()
    let encoder = commandBuf.blitCommandEncoder()
    for i in 0..<src.count {
        encoder.copyFromTexture(src[i], sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(), sourceSize: MTLSizeMake(dest.width, dest.height, 1), toTexture: dest, destinationSlice: i, destinationLevel: 0, destinationOrigin: MTLOrigin())
    }
    encoder.endEncoding()
    commandBuf.commit()
    commandBuf.waitUntilCompleted()

    if let err = commandBuf.error {
        fatalError("MetalExecutionError: " + err.description)
    }
}

if Process.arguments.count >= 2 {
    let path = Process.arguments[1]

    let resizeJsonFile = "scale2.0x_model.json"
    let jsonObj = try! NSJSONSerialization.JSONObjectWithData(NSData(contentsOfFile: resizeJsonFile)!, options: []) as! [[String: AnyObject]]
    let model = jsonObj.map(createModelLayer)

    let image = loadImage(path)!
    let width = CGImageGetWidth(image)
    let height = CGImageGetHeight(image)
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
