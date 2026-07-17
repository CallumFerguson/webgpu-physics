import { createServer } from "vite";

const port = 4173;

export default async function globalSetup(): Promise<() => Promise<void>> {
  const server = await createServer({
    server: {
      host: "127.0.0.1",
      port,
      strictPort: true,
      // Deterministic E2E pages reload frequently but never hot-reload source.
      // Avoid retaining an unnecessary WebSocket for every browser lifecycle.
      hmr: false,
    },
  });
  await server.listen();

  return async () => {
    await server.close();
  };
}
