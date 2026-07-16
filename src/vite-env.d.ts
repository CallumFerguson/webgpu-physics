/// <reference types="vite/client" />

interface Window {
  __webgpuRenderDone?: Promise<void>;
}
