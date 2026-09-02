# 介译图形工作台

基于 React、TypeScript 和 vinext 的本地翻译界面，与项目根目录中的 Python API 配合使用。完整安装和功能说明见 [主 README](../README.md)。

## 本地开发

需要 Node.js 22.13+ 和 npm。先在项目根目录启动 API：

```bash
uv sync --locked --extra api
uv run uvicorn jieyi.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

在另一个终端中进入 `web/`：

```bash
npm ci
npx vinext dev --host 127.0.0.1 --port 3000
```

打开 `http://localhost:3000`。默认 API 地址为 `http://127.0.0.1:8000`。

自定义 API 地址时，在 `web/.env.local` 中设置以下非敏感配置，再重启开发服务或重新构建：

```dotenv
NEXT_PUBLIC_JIEYI_API=http://127.0.0.1:8001
```

`NEXT_PUBLIC_*` 会进入浏览器代码，不能用于保存模型密钥。模型连接和密钥由本地 Python 服务管理。

## 构建与检查

在 macOS / Linux 终端中执行：

```bash
npm test
npm run lint
npx tsc --noEmit
npm run start -- --port 3000
```

`npm test` 会先生产构建，再执行渲染和阅读导航测试。当前 npm 的构建脚本使用 POSIX 环境变量语法；Windows 开发者可在 Git Bash / WSL 中使用这些脚本，或在 PowerShell 中直接执行：

```powershell
npx vinext build
node --experimental-strip-types --test tests/*.test.mjs
npx vinext start --port 3000
```

修改 `NEXT_PUBLIC_JIEYI_API` 后需重新构建生产版本。单独启动网页不会启动 Python API。

## 目录

- `app/`：书库、导入、阅读、翻译、模型设置及术语审核界面。
- `tests/`：生产渲染和阅读导航回归测试。
- `public/`：网页静态资源。
- `worker/`、`.openai/hosting.json`：现有 Cloudflare / Sites 构建入口与可选绑定声明。
- `db/`、`examples/d1/`：可选 D1 示例；介译业务数据由 Python API 保存到本机 SQLite。

本地开发不需要配置 D1、R2 或部署账户。当前界面默认访问浏览器所在电脑上的 API；公开部署网页不会自动托管 Python 服务或书籍数据库。

## 许可证

介译原创代码采用 [MIT](../LICENSE) 许可证，第三方说明见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
