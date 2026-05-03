import compositeShader from './shaders/composite.wgsl?raw'
import maskRefineShader from './shaders/maskRefine.wgsl?raw'
import plateInitShader from './shaders/plateInit.wgsl?raw'
import plateLearnShader from './shaders/plateLearn.wgsl?raw'
import pushDownShader from './shaders/pushDown.wgsl?raw'
import pushUpShader from './shaders/pushUp.wgsl?raw'
import pyramidBaseShader from './shaders/pyramidBase.wgsl?raw'
import { RENDER_HEIGHT, RENDER_WIDTH, type DebugView, type FxSettings } from './types'

const WORKGROUP_SIZE = 8
const MASK_WIDTH = 256
const MASK_HEIGHT = 256
const PYRAMID_SIZES = [
  [960, 540],
  [480, 270],
  [240, 135],
  [120, 68],
  [60, 34],
  [30, 17],
] as const

export class WebGpuFxRenderer {
  private adapter: GPUAdapter | null = null
  private device: GPUDevice | null = null
  private context: GPUCanvasContext | null = null
  private format: GPUTextureFormat = 'bgra8unorm'
  private sampler: GPUSampler | null = null
  private videoTexture: GPUTexture | null = null
  private rawMaskTexture: GPUTexture | null = null
  private refinedMaskTextures: GPUTexture[] = []
  private plateTextures: GPUTexture[] = []
  private pyramidTextures: GPUTexture[] = []
  private pyramidScratchTextures: GPUTexture[] = []
  private maskUniformBuffer: GPUBuffer | null = null
  private plateInitUniformBuffer: GPUBuffer | null = null
  private plateUniformBuffer: GPUBuffer | null = null
  private compositeUniformBuffer: GPUBuffer | null = null
  private pyramidSizeBuffers: GPUBuffer[] = []
  private maskRefinePipeline: GPUComputePipeline | null = null
  private plateInitPipeline: GPUComputePipeline | null = null
  private plateLearnPipeline: GPUComputePipeline | null = null
  private pyramidBasePipeline: GPUComputePipeline | null = null
  private pushDownPipeline: GPUComputePipeline | null = null
  private pushUpPipeline: GPUComputePipeline | null = null
  private compositePipeline: GPURenderPipeline | null = null
  private maskRefineBindGroups: GPUBindGroup[] = []
  private plateInitBindGroups: GPUBindGroup[] = []
  private plateLearnBindGroups: GPUBindGroup[] = []
  private pyramidBaseBindGroups: GPUBindGroup[] = []
  private pushDownBindGroups: GPUBindGroup[] = []
  private pushUpBindGroups: GPUBindGroup[] = []
  private compositeBindGroups: GPUBindGroup[] = []
  private refinedMaskIndex = 0
  private plateIndex = 0
  private plateInitialized = false
  private emptyMask = new Uint8Array(MASK_WIDTH * MASK_HEIGHT * 4)

  get label(): string {
    return this.adapter?.info?.description || this.adapter?.info?.vendor || 'WebGPU'
  }

  async init(canvas: HTMLCanvasElement): Promise<void> {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.')

    this.adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!this.adapter) throw new Error('No WebGPU adapter was found.')

    this.device = await this.adapter.requestDevice()
    this.context = canvas.getContext('webgpu')
    if (!this.context) throw new Error('The canvas does not expose a WebGPU context.')

    this.format = navigator.gpu.getPreferredCanvasFormat()
    canvas.width = RENDER_WIDTH
    canvas.height = RENDER_HEIGHT
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'opaque',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
    })

    this.sampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    })

    this.maskUniformBuffer = this.createUniformBuffer(8, 'mask-refine-uniforms')
    this.plateInitUniformBuffer = this.createUniformBuffer(4, 'plate-init-uniforms')
    this.plateUniformBuffer = this.createUniformBuffer(4, 'plate-learn-uniforms')
    this.compositeUniformBuffer = this.createUniformBuffer(16, 'composite-uniforms')
    this.pyramidSizeBuffers = PYRAMID_SIZES.map(([width, height], index) => {
      const buffer = this.createUniformBuffer(4, `pyramid-size-${index}`)
      this.device!.queue.writeBuffer(buffer, 0, new Float32Array([width, height, 0, 0]))
      return buffer
    })

    this.createTextures()
    this.createPipelines()
    this.uploadEmptyMask()
  }

  resetPlate(): void {
    this.plateInitialized = false
  }

  render(
    source: HTMLCanvasElement,
    maskSource: OffscreenCanvas | HTMLCanvasElement | null,
    settings: FxSettings,
    time: number,
  ): void {
    if (!this.ready()) return

    const device = this.device!
    const context = this.context!
    device.queue.copyExternalImageToTexture(
      { source },
      { texture: this.videoTexture! },
      { width: RENDER_WIDTH, height: RENDER_HEIGHT },
    )
    if (maskSource) {
      device.queue.copyExternalImageToTexture(
        { source: maskSource },
        { texture: this.rawMaskTexture! },
        { width: MASK_WIDTH, height: MASK_HEIGHT },
      )
    }
    this.writeMaskUniforms(settings)
    this.writePlateInitUniforms()
    this.writePlateUniforms(settings)
    this.writeCompositeUniforms(settings, time)

    const encoder = device.createCommandEncoder()
    if (!this.plateInitialized) {
      const initPass = encoder.beginComputePass({ label: 'plate-init' })
      initPass.setPipeline(this.plateInitPipeline!)
      for (const bindGroup of this.plateInitBindGroups) {
        initPass.setBindGroup(0, bindGroup)
        initPass.dispatchWorkgroups(Math.ceil(RENDER_WIDTH / WORKGROUP_SIZE), Math.ceil(RENDER_HEIGHT / WORKGROUP_SIZE))
      }
      initPass.end()
      this.plateInitialized = true
    }

    const maskPass = encoder.beginComputePass({ label: 'mask-refine' })
    maskPass.setPipeline(this.maskRefinePipeline!)
    maskPass.setBindGroup(0, this.maskRefineBindGroups[this.refinedMaskIndex])
    maskPass.dispatchWorkgroups(Math.ceil(RENDER_WIDTH / WORKGROUP_SIZE), Math.ceil(RENDER_HEIGHT / WORKGROUP_SIZE))
    maskPass.end()
    this.refinedMaskIndex = 1 - this.refinedMaskIndex

    const platePass = encoder.beginComputePass({ label: 'plate-learn' })
    platePass.setPipeline(this.plateLearnPipeline!)
    const plateLearnGroup = this.plateIndex * 2 + this.refinedMaskIndex
    platePass.setBindGroup(0, this.plateLearnBindGroups[plateLearnGroup])
    platePass.dispatchWorkgroups(Math.ceil(RENDER_WIDTH / WORKGROUP_SIZE), Math.ceil(RENDER_HEIGHT / WORKGROUP_SIZE))
    platePass.end()
    this.plateIndex = 1 - this.plateIndex

    this.encodeInpaintPasses(encoder)

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.04, g: 0.05, b: 0.055, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    })
    renderPass.setPipeline(this.compositePipeline!)
    renderPass.setBindGroup(0, this.compositeBindGroups[this.plateIndex * 2 + this.refinedMaskIndex])
    renderPass.draw(3)
    renderPass.end()

    device.queue.submit([encoder.finish()])
  }

  dispose(): void {
    this.videoTexture?.destroy()
    this.rawMaskTexture?.destroy()
    this.refinedMaskTextures.forEach((texture) => texture.destroy())
    this.plateTextures.forEach((texture) => texture.destroy())
    this.pyramidTextures.forEach((texture) => texture.destroy())
    this.pyramidScratchTextures.forEach((texture) => texture.destroy())
    this.maskUniformBuffer?.destroy()
    this.plateInitUniformBuffer?.destroy()
    this.plateUniformBuffer?.destroy()
    this.compositeUniformBuffer?.destroy()
    this.pyramidSizeBuffers.forEach((buffer) => buffer.destroy())
    this.maskRefineBindGroups = []
    this.plateInitBindGroups = []
    this.plateLearnBindGroups = []
    this.pyramidBaseBindGroups = []
    this.pushDownBindGroups = []
    this.pushUpBindGroups = []
    this.compositeBindGroups = []
  }

  private createUniformBuffer(floatCount: number, label: string): GPUBuffer {
    return this.device!.createBuffer({
      size: floatCount * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label,
    })
  }

  private ready(): boolean {
    return Boolean(
      this.device &&
        this.context &&
        this.videoTexture &&
        this.rawMaskTexture &&
        this.refinedMaskTextures.length === 2 &&
        this.plateTextures.length === 2 &&
        this.pyramidTextures.length === PYRAMID_SIZES.length &&
        this.pyramidScratchTextures.length === PYRAMID_SIZES.length &&
        this.maskRefineBindGroups.length === 2 &&
        this.plateInitBindGroups.length === 2 &&
        this.plateLearnBindGroups.length === 4 &&
        this.pyramidBaseBindGroups.length === 4 &&
        this.pushDownBindGroups.length === PYRAMID_SIZES.length - 1 &&
        this.pushUpBindGroups.length === PYRAMID_SIZES.length - 1 &&
        this.compositeBindGroups.length === 4,
    )
  }

  private createTextures(): void {
    const base: GPUTextureDescriptor = {
      size: { width: RENDER_WIDTH, height: RENDER_HEIGHT },
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.RENDER_ATTACHMENT,
    }
    this.videoTexture = this.device!.createTexture({ ...base, label: 'current-camera-frame' })
    this.rawMaskTexture = this.device!.createTexture({
      ...base,
      size: { width: MASK_WIDTH, height: MASK_HEIGHT },
      label: 'raw-mediapipe-mask',
    })
    this.refinedMaskTextures = [
      this.device!.createTexture({ ...base, label: 'refined-mask-a' }),
      this.device!.createTexture({ ...base, label: 'refined-mask-b' }),
    ]
    this.plateTextures = [
      this.device!.createTexture({ ...base, label: 'temporal-plate-a' }),
      this.device!.createTexture({ ...base, label: 'temporal-plate-b' }),
    ]
    this.pyramidTextures = PYRAMID_SIZES.map(([width, height], index) =>
      this.device!.createTexture({ ...base, size: { width, height }, label: `inpaint-pyramid-${index}` }),
    )
    this.pyramidScratchTextures = PYRAMID_SIZES.map(([width, height], index) =>
      this.device!.createTexture({ ...base, size: { width, height }, label: `inpaint-pyramid-scratch-${index}` }),
    )
  }

  private createPipelines(): void {
    this.maskRefinePipeline = this.device!.createComputePipeline({
      layout: 'auto',
      compute: { module: this.device!.createShaderModule({ code: maskRefineShader }), entryPoint: 'main' },
    })
    this.plateInitPipeline = this.device!.createComputePipeline({
      layout: 'auto',
      compute: { module: this.device!.createShaderModule({ code: plateInitShader }), entryPoint: 'main' },
    })
    this.plateLearnPipeline = this.device!.createComputePipeline({
      layout: 'auto',
      compute: { module: this.device!.createShaderModule({ code: plateLearnShader }), entryPoint: 'main' },
    })
    this.pyramidBasePipeline = this.device!.createComputePipeline({
      layout: 'auto',
      compute: { module: this.device!.createShaderModule({ code: pyramidBaseShader }), entryPoint: 'main' },
    })
    this.pushDownPipeline = this.device!.createComputePipeline({
      layout: 'auto',
      compute: { module: this.device!.createShaderModule({ code: pushDownShader }), entryPoint: 'main' },
    })
    this.pushUpPipeline = this.device!.createComputePipeline({
      layout: 'auto',
      compute: { module: this.device!.createShaderModule({ code: pushUpShader }), entryPoint: 'main' },
    })
    const compositeModule = this.device!.createShaderModule({ code: compositeShader })
    this.compositePipeline = this.device!.createRenderPipeline({
      layout: 'auto',
      vertex: { module: compositeModule, entryPoint: 'vertexMain' },
      fragment: {
        module: compositeModule,
        entryPoint: 'fragmentMain',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list' },
    })
    this.createBindGroups()
  }

  private createBindGroups(): void {
    const videoView = this.videoTexture!.createView()
    const rawMaskView = this.rawMaskTexture!.createView()
    const refinedViews = this.refinedMaskTextures.map((texture) => texture.createView())
    const plateViews = this.plateTextures.map((texture) => texture.createView())
    const pyramidViews = this.pyramidTextures.map((texture) => texture.createView())
    const scratchViews = this.pyramidScratchTextures.map((texture) => texture.createView())

    this.maskRefineBindGroups = [
      this.device!.createBindGroup({
        layout: this.maskRefinePipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: rawMaskView },
          { binding: 1, resource: videoView },
          { binding: 2, resource: refinedViews[0] },
          { binding: 3, resource: refinedViews[1] },
          { binding: 4, resource: { buffer: this.maskUniformBuffer! } },
          { binding: 5, resource: this.sampler! },
        ],
      }),
      this.device!.createBindGroup({
        layout: this.maskRefinePipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: rawMaskView },
          { binding: 1, resource: videoView },
          { binding: 2, resource: refinedViews[1] },
          { binding: 3, resource: refinedViews[0] },
          { binding: 4, resource: { buffer: this.maskUniformBuffer! } },
          { binding: 5, resource: this.sampler! },
        ],
      }),
    ]

    this.plateInitBindGroups = plateViews.map((plateView) =>
      this.device!.createBindGroup({
        layout: this.plateInitPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: videoView },
          { binding: 1, resource: plateView },
          { binding: 2, resource: { buffer: this.plateInitUniformBuffer! } },
        ],
      }),
    )

    this.plateLearnBindGroups = [0, 1].flatMap((plateIndex) =>
      [0, 1].map((maskIndex) =>
        this.device!.createBindGroup({
          layout: this.plateLearnPipeline!.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: videoView },
            { binding: 1, resource: plateViews[plateIndex] },
            { binding: 2, resource: refinedViews[maskIndex] },
            { binding: 3, resource: plateViews[1 - plateIndex] },
            { binding: 4, resource: { buffer: this.plateUniformBuffer! } },
          ],
        }),
      ),
    )

    this.pyramidBaseBindGroups = [0, 1].flatMap((plateIndex) =>
      [0, 1].map((maskIndex) =>
        this.device!.createBindGroup({
        layout: this.pyramidBasePipeline!.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: videoView },
            { binding: 1, resource: plateViews[plateIndex] },
            { binding: 2, resource: refinedViews[maskIndex] },
            { binding: 3, resource: pyramidViews[0] },
            { binding: 4, resource: { buffer: this.pyramidSizeBuffers[0] } },
          ],
        }),
      ),
    )

    this.pushDownBindGroups = PYRAMID_SIZES.slice(1).map((_, index) =>
      this.device!.createBindGroup({
        layout: this.pushDownPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: pyramidViews[index] },
          { binding: 1, resource: pyramidViews[index + 1] },
          { binding: 2, resource: { buffer: this.pyramidSizeBuffers[index + 1] } },
        ],
      }),
    )

    this.pushUpBindGroups = PYRAMID_SIZES.slice(0, -1).map((_, index) => {
      const level = PYRAMID_SIZES.length - 2 - index
      return this.device!.createBindGroup({
        layout: this.pushUpPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: pyramidViews[level] },
          { binding: 1, resource: pyramidViews[level + 1] },
          { binding: 2, resource: scratchViews[level] },
          { binding: 3, resource: this.sampler! },
          { binding: 4, resource: { buffer: this.pyramidSizeBuffers[level] } },
        ],
      })
    })

    this.compositeBindGroups = [0, 1].flatMap((plateIndex) =>
      [0, 1].map((maskIndex) =>
        this.device!.createBindGroup({
          layout: this.compositePipeline!.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: this.sampler! },
            { binding: 1, resource: videoView },
            { binding: 2, resource: plateViews[plateIndex] },
            { binding: 3, resource: pyramidViews[0] },
            { binding: 4, resource: refinedViews[maskIndex] },
            { binding: 5, resource: { buffer: this.compositeUniformBuffer! } },
          ],
        }),
      ),
    )
  }

  private encodeInpaintPasses(encoder: GPUCommandEncoder): void {
    const basePass = encoder.beginComputePass({ label: 'inpaint-base' })
    basePass.setPipeline(this.pyramidBasePipeline!)
    basePass.setBindGroup(0, this.pyramidBaseBindGroups[this.plateIndex * 2 + this.refinedMaskIndex])
    basePass.dispatchWorkgroups(Math.ceil(RENDER_WIDTH / WORKGROUP_SIZE), Math.ceil(RENDER_HEIGHT / WORKGROUP_SIZE))
    basePass.end()

    for (let i = 0; i < this.pushDownBindGroups.length; i += 1) {
      const [width, height] = PYRAMID_SIZES[i + 1]
      const pass = encoder.beginComputePass({ label: `push-down-${i}` })
      pass.setPipeline(this.pushDownPipeline!)
      pass.setBindGroup(0, this.pushDownBindGroups[i])
      pass.dispatchWorkgroups(Math.ceil(width / WORKGROUP_SIZE), Math.ceil(height / WORKGROUP_SIZE))
      pass.end()
    }

    for (let i = 0; i < this.pushUpBindGroups.length; i += 1) {
      const level = PYRAMID_SIZES.length - 2 - i
      const [width, height] = PYRAMID_SIZES[level]
      const pass = encoder.beginComputePass({ label: `push-up-${level}` })
      pass.setPipeline(this.pushUpPipeline!)
      pass.setBindGroup(0, this.pushUpBindGroups[i])
      pass.dispatchWorkgroups(Math.ceil(width / WORKGROUP_SIZE), Math.ceil(height / WORKGROUP_SIZE))
      pass.end()
      encoder.copyTextureToTexture(
        { texture: this.pyramidScratchTextures[level] },
        { texture: this.pyramidTextures[level] },
        { width, height },
      )
    }
  }

  private uploadEmptyMask(): void {
    this.device!.queue.writeTexture(
      { texture: this.rawMaskTexture! },
      this.emptyMask,
      { bytesPerRow: MASK_WIDTH * 4, rowsPerImage: MASK_HEIGHT },
      { width: MASK_WIDTH, height: MASK_HEIGHT },
    )
  }

  private writeMaskUniforms(settings: FxSettings): void {
    this.device!.queue.writeBuffer(
      this.maskUniformBuffer!,
      0,
      new Float32Array([
        RENDER_WIDTH,
        RENDER_HEIGHT,
        settings.maskFeather,
        settings.maskStability,
        settings.followLock,
        0,
        0,
        0,
      ]),
    )
  }

  private writePlateUniforms(settings: FxSettings): void {
    this.device!.queue.writeBuffer(
      this.plateUniformBuffer!,
      0,
      new Float32Array([RENDER_WIDTH, RENDER_HEIGHT, settings.plateLearning, 0]),
    )
  }

  private writePlateInitUniforms(): void {
    this.device!.queue.writeBuffer(
      this.plateInitUniformBuffer!,
      0,
      new Float32Array([RENDER_WIDTH, RENDER_HEIGHT, 0.25, 0]),
    )
  }

  private writeCompositeUniforms(settings: FxSettings, time: number): void {
    this.device!.queue.writeBuffer(
      this.compositeUniformBuffer!,
      0,
      new Float32Array([
        RENDER_WIDTH,
        RENDER_HEIGHT,
        time * 0.001,
        settings.opacity,
        settings.jelly,
        settings.water,
        settings.cloth,
        settings.refraction,
        settings.edgeGain,
        debugViewToNumber(settings.debugView),
        settings.inpaintFallback,
        0,
        0,
        0,
        0,
        0,
      ]),
    )
  }
}

function debugViewToNumber(view: DebugView): number {
  if (view === 'matte') return 1
  if (view === 'plate') return 2
  if (view === 'inpaint') return 3
  return 0
}
