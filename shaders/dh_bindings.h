/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// nvpro convention for descriptorset bindings
// "dh" = "device--host"

#ifndef BINDINGS_H
#define BINDINGS_H

// Descriptor set index for raytracing pipeline
const int DSRt = 0;

const int BRtTlas                   = 0;
const int BRtOutBaseColor_Metalness = 1;
const int BRtOutSpecAlbedo          = 2;
const int BRtOutSpecHitDist         = 3;
const int BRtOutNormalRoughness     = 4;
const int BRtOutMotionVectors       = 5;
const int BRtOutViewZ               = 6;
const int BRtOutColor               = 7;
const int BRtFrameInfo              = 8;
const int BRtSkyParam               = 9;
const int BRtTextures               = 10;

#endif  // !BINDINGS_H
