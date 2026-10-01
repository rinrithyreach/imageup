# Real-ESRGAN — model licence and modification notice

The file `realesr-general-x4v3.onnx` in this folder is the Real-ESRGAN
"general x4v3" model (SRVGGNetCompact), by Xintao Wang and contributors, used
under the BSD 3-Clause licence reproduced below.

## Modification

This copy differs from the upstream export by **nine bytes**, none of which are
weights. The upstream file declares the graph's output with the same symbolic
dimension names as its input:

    input   batch_size, 3, height, width
    output  batch_size, 3, height, width     <- wrong: the output is 4x larger

Because the two shapes share the names `height` and `width`, ONNX Runtime
infers that the output is the same size as the input and pre-allocates a
buffer to match. The WebGPU backend then refuses to run at all:

    Shape mismatch attempting to re-use buffer. {1,90,120,3} != {1,360,480,3}

The output's two dimension names were renamed so that they are distinct
symbols, which lets the runtime size the output at run time instead of
guessing it from the input:

    output  batch_size, 3, out_ht, out_w

Only those names changed, and only on the output declaration. The weights, the
graph and the arithmetic are untouched: WebGPU and WebAssembly produce
identical numbers from the corrected file, which was verified before shipping.

To reproduce the change from an upstream copy, with the replacement names
chosen to be exactly as long as the originals so that nothing else shifts:

    const fs = require('fs');
    const b = fs.readFileSync('model.onnx');
    b.write('out_ht', b.lastIndexOf('height'), 'latin1');
    b.write('out_w', b.lastIndexOf('width'), 'latin1');
    fs.writeFileSync('model.onnx', b);

---

BSD 3-Clause License

Copyright (c) 2021, Xintao Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
