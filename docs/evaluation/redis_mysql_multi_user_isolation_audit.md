# Redis / MySQL 多用户隔离现状审计

## 结论

| 能力 | 当前结论 |
|---|---|
| 数据库多用户逻辑隔离 | **已实现，但实际跑在 SQLite** |
| MySQL 接入 | **未实现** |
| Redis 接入 | **有封装，默认关闭，当前 uv 依赖也未包含 Redis 客户端** |
| Redis 严格按用户隔离 | **未完整实现** |
| tenant 逻辑已删除 | **没有；代码和表结构中仍广泛存在 `tenant_id`** |

所以目前准确的项目表述应是：“基于 SQLAlchemy 在应用层按 tenant/user/session 过滤，实现 SQLite 下的多用户会话隔离；Redis 和 MySQL 仍是工程脚手架或待接入项。”不能表述成“已基于 MySQL + Redis 完成生产级隔离”。

## 证据

### 数据库

- `DATABASE_URL` 为空时默认使用 SQLite；`docker-compose.yml` 没有 MySQL 服务。
- `pyproject.toml` 没有 `pymysql`、`asyncmy` 或 `aiomysql`，因此仅设置 MySQL URL 也无法直接运行。
- 消息、心电报告和用户查询确实同时过滤 `tenant_id`、`user_id`，会话接口还过滤 `session_id`。
- 用户唯一约束是 `(tenant_id, user_id)`，不是全局 `user_id` 唯一；如果决定取消 tenant，必须迁移表结构与唯一约束，不能只在接口层忽略该字段。

### Redis

- `REDIS_ENABLED=false`，连接失败后退化为进程内字典；多实例之间不共享，也不具备持久性。
- 限流 identity 包含 `tenant:user:IP`，具备逻辑隔离。
- 语义缓存指纹包含 tenant，但不包含 user；同一 tenant 的用户会共享低风险答案，不能称为严格用户隔离。
- 任务状态 key 只有随机 `job_id`；查询接口会再校验 payload 中的 tenant/user，属于应用层鉴权，而不是 key 空间隔离。
- 后台任务实际是进程内 `ThreadPoolExecutor`，不是 Redis 队列，也不具备宕机恢复能力。
- `requirements.txt` 声明了 `redis`，但 uv 使用的 `pyproject.toml` 没有，依赖来源不一致。

## 若只保留“多用户”，最小修正方向

1. 把 `user_id` 设为全局唯一身份，所有表和缓存 key 统一使用 `user_id`；做数据库迁移后再删除 `tenant_id`。
2. MySQL 使用正式 migration 工具创建唯一索引和联合索引，例如消息表 `(user_id, session_id, timestamp)`。
3. Redis key 统一为 `mg:{feature}:{user_id}:...`；任务查询仍保留服务端 owner 校验，避免只凭 job ID 读取。
4. 增加真实 MySQL/Redis 集成测试：用户 A 不得读取、删除、命中用户 B 的会话、报告、任务或个性化缓存。

以上为代码静态审计；本轮没有启动 MySQL 或 Redis，因为仓库当前没有对应 compose 服务和完整 uv 依赖。
