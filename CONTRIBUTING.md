# 贡献指南

欢迎改进介译的翻译流程、EPUB 保真、术语发现、阅读体验和文档。较大的功能变更建议先在 [Issues](https://github.com/shengbaiwang/jieyi/issues) 中描述问题和预期行为。

## 准备环境

1. Fork 仓库并克隆到本地，从 `main` 创建工作分支。
2. 按 [README](README.md) 安装 Python 和网页依赖。
3. 使用 `examples/sample.md` 或自行编写的短文本验证；不要提交私人书稿、真实密钥或本地数据库。

## 验证改动

Python 核心和 API：

```bash
uv sync --locked --extra api --extra dev
uv run pytest
uv run ruff check src tests
```

网页端（macOS / Linux / Git Bash / WSL）：

```bash
cd web
npm ci
npm test
npm run lint
npx tsc --noEmit
```

Windows PowerShell 的直接构建命令见 [网页开发指南](web/README.md)。GitHub Actions 会运行 Python 测试和网页构建、测试、静态检查。

## 提交 Pull Request

- 描述用户遇到的问题、改动后的行为和实际运行的验证。
- 行为变更应包含能复现问题或验证新行为的测试；纯文档和格式修正无需新增测试。
- 界面改动可附截图；涉及书籍内容时使用自写示例或可以公开分享的内容。
- 维护清晰的领域、持久化、供应商适配器和 API 边界，详见 [架构说明](docs/ARCHITECTURE.md)。
- 引入外部代码或依赖时保留必要的版权和许可证声明，并更新 [第三方说明](THIRD_PARTY_NOTICES.md)。

提交到本项目的原创贡献按仓库的 [MIT 许可证](LICENSE) 分发。
