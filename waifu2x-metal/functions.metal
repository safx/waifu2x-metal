//
//  functions.metal
//  waifu2x-metal
//
//  Created by Safx Developer on 2015/06/20.
//  Copyright © 2015年 Safx Developers. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void splitToRGBChannels(texture2d<float, access::read> in[[texture(0)]],
                               texture2d_array<float, access::write> outRGB[[texture(1)]],
                               uint2 gid[[thread_position_in_grid]])
{
    if (gid.x < in.get_width() && gid.y < in.get_height()) {
        outRGB.write(float4(in.read(gid).r, 0.0f, 0.0f, 0.0f), gid, 0);
        outRGB.write(float4(in.read(gid).g, 0.0f, 0.0f, 0.0f), gid, 1);
        outRGB.write(float4(in.read(gid).b, 0.0f, 0.0f, 0.0f), gid, 2);
    }
}

kernel void combineRGBChannels(texture2d_array<float, access::read> in[[texture(0)]],
                               texture2d<float, access::write> out[[texture(1)]],
                               uint2 gid[[thread_position_in_grid]])
{
    if (gid.x < in.get_width() && gid.y < in.get_height()) {
        float4 outColor(in.read(gid, 0).r, in.read(gid, 1).r, in.read(gid, 2).r, 1.0f);
        out.write(outColor, gid);
    }
}

kernel void waifu2x(texture2d_array<float, access::read> in[[texture(0)]],
                            texture2d<float, access::write> out[[texture(1)]],
                            constant float3x3* weights[[buffer(0)]],
                            constant float&    bias[[buffer(1)]],
                            uint2 gid[[thread_position_in_grid]])
{
    if (gid.x >= in.get_width() || gid.y >= in.get_height()) {
        return;
    }

    int2 m00 = int2(-1, -1);
    int2 m10 = int2( 0, -1);
    int2 m20 = int2(+1, -1);
    int2 m01 = int2(-1,  0);
    int2 m11 = int2( 0,  0);
    int2 m21 = int2(+1,  0);
    int2 m02 = int2(-1, +1);
    int2 m12 = int2( 0, +1);
    int2 m22 = int2(+1, +1);

    if (1 <= gid.x && gid.x < in.get_width() - 1
        && 1 <= gid.y && gid.y < in.get_height() - 1) {
        // fallthrough
    } else {
        if (gid.x == 0) {
            m00 = m10;
            m01 = m11;
            m02 = m12;
        } else if (gid.x == in.get_width() - 1) {
            m20 = m10;
            m21 = m11;
            m22 = m12;
        }

        if (gid.y == 0) {
            m00 = m01;
            m10 = m11;
            m20 = m21;
        } else if (gid.y == in.get_height() - 1) {
            m02 = m01;
            m12 = m11;
            m22 = m21;
        }
    }

    float partial = bias;
    for (uint i = 0; i < in.get_array_size(); ++i) {
        float3 in0 = float3(in.read(gid + uint2(m00), i).r,
                            in.read(gid + uint2(m10), i).r,
                            in.read(gid + uint2(m20), i).r);

        float3 in1 = float3(in.read(gid + uint2(m01), i).r,
                            in.read(gid + uint2(m11), i).r,
                            in.read(gid + uint2(m21), i).r);

        float3 in2 = float3(in.read(gid + uint2(m02), i).r,
                            in.read(gid + uint2(m12), i).r,
                            in.read(gid + uint2(m22), i).r);

        float3x3 weight = weights[i];
        partial += dot(in0, weight[0])
                 + dot(in1, weight[1])
                 + dot(in2, weight[2]);
    }

    float p = fmax(partial, 0) + 0.1 * fmin(partial, 0);
    float4 outColor(p, 0, 0, 0);
    out.write(outColor, gid);
}
