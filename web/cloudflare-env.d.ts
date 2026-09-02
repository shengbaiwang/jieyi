// Only the binding used by the optional D1 adapter is exposed here.
// Keep Worker globals out of the browser application's DOM type environment.
declare module "cloudflare:workers" {
  export const env: {
    DB?: import("@cloudflare/workers-types").D1Database;
  };
}
