# The web tier: Next UI, /api handlers, the /ws upgrade and the MJPEG relay, all
# in one process. See server.ts.
#
# No Python here. Inference lives in worker/Dockerfile, out of process, because
# a CPU-pinned decode loop in this container would stall the dashboard.
FROM node:22-slim

WORKDIR /app

# Dependencies first: this layer is rebuilt only when the lockfile moves.
COPY package.json package-lock.json* ./
# Dev dependencies are KEPT. `next build` needs typescript, and server.ts runs
# through tsx rather than being compiled ahead of time — the same entry point in
# production as in development, which is one fewer thing that can differ.
RUN npm ci

COPY . .
RUN npm run build

ENV NODE_ENV=production
ENV PORT=3000
EXPOSE 3000

# No HEALTHCHECK curl: this image has no curl and adding one to run a health
# probe is a package to patch forever. Compose checks it with node instead.
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
  CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||3000)+'/api/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["npm", "start"]
