# 网页改版与维护

这是原 GitHub Pages 网页的完整静态改版。保留原始 `README.md` 作为文献来源，无需安装第三方运行依赖。

## 已实现

- 浅色学术风格、可记忆的深色模式、桌面与移动端布局。
- 正文、导航、研究卡片与文献卡片字号增大；Research Map 和 Field Notes 前置，The Reading Room 位于最后一个主内容区。
- 首页按晴 → 雾 → 雨 → 雪自动循环，每种天气停留 8 秒；天气标签、进度条、相机短片和点云粒子同步切换。
- 右上角播放真实数据集相机短片：晴、雨、雪来自 Boreas，雾来自 RADIATE。四段静音循环视频合计约 1.4 MB；提供来源、序列与授权说明，加载失败时保留对应天气的实拍首帧和来源链接。
- 点云示意保留持续移动的道路、扫描光带、雷达转动、目标体积重建和鼠标视差。
- 统一按钮暂停／恢复天气轮播、相机短片和点云动画，并记住播放状态。首次访问遵循系统减少动画设置；系统设置变化时重新遵循系统偏好。页面不可见或画面离开视口时停止播放，以限制资源消耗。
- 192 个文献／资源条目，完整保留 README 中 213 条分类记录和 265 个不同的文献／代码／数据／项目链接（截至 2026 年 10 月 12 日）。
- 10 个研究方向，标题／别名／方向／期刊会议搜索，年份与资源类型组合筛选，排序与渐进加载。
- 浏览器本地阅读清单、Markdown 导出、筛选链接分享、前进后退状态恢复。
- 可点击研究地图与键盘可操作的传感器标签页，移动端折叠菜单。
- 本地字体，避免页面依赖外部字体服务；图片延迟加载。原 23.8 MB 动图不再作为页面依赖。
- 键盘 `/` 聚焦搜索、Esc 清空搜索、跳过导航、可见焦点、动态结果播报和空结果提示。

## GitHub Pages 使用方式

GitHub Pages 从 `main` 分支的仓库根目录发布，项目主页为 <https://ylwhxht.github.io/Awesome-3D_Object_Detection_in_Adverse_Weather/>。所有资源均使用相对路径，适配这一仓库子路径。

发布所需文件为 `index.html`、`styles.css`、`app.js`、`scene.js`、`data.js`、`favicon.svg`、`.nojekyll`、`fonts/`、`images/kradar_examples.png`、四组 `images/camera-*.mp4`／`.jpg` 和 `images/CAMERA-SOURCES.md`。`dist/` 是相同页面的打包输出，供其他静态托管平台使用。

改版以提交 `8086a29b4a17bffc0669f821d21ac4a51bd8ac57` 为基线，保留原始文献清单和图片。后续提交到 `main` 后，由 GitHub Pages 发布流程更新主页。

## 更新文献

1. 按原格式维护 `README.md` 中的论文、年份、分类和链接。
2. 执行 `node scripts/build.mjs`，重新生成 `data.js`、`index.html` 中的资源版本标识和 `dist/`。
3. 执行 `node --check app.js`、`node --check scene.js`、`node scripts/check.mjs`。
4. 上传或提交更新后的静态文件，包括 `index.html` 和 `data.js`。资源版本随内容变化，避免浏览器继续使用上一版脚本、样式或文献数据。

解析器只合并标题、年份和主要来源链接相同的记录，并保留所有分类与补充链接；不同版本和来源中不同标题仍然独立保留。年份优先采用每条记录的出版信息，再回退到分组年份。

补充的 arXiv 和项目链接按原清单保留；标为 `project (code pending)` 的链接不会计入“已有代码”筛选和统计。出版信息同时注明在线年和卷年时，筛选采用卷年，卡片保留完整出版说明。

本次为网页改版，文献元数据与出版信息继承原清单，未进行逐篇文献真实性或出版信息审计。点云画面是交互示意图，相机窗口是真实数据集画面；两者只对应天气类别，不代表同步传感器测量或实际检测输出。

## 本地预览

需要 Node.js。先执行 `node scripts/build.mjs`，再执行 `node scripts/serve.mjs`，打开 `http://127.0.0.1:4173`。

## 素材来源

- 文献与研究分类：原 [Awesome-3D_Object_Detection_in_Adverse_Weather 仓库](https://github.com/ylwhxht/Awesome-3D_Object_Detection_in_Adverse_Weather)。
- 天气样例图片：原仓库中的 K-Radar 图片，网页保留 K-Radar 来源链接。
- 相机短片：Boreas（CC BY 4.0）与 RADIATE（CC BY-NC-SA 4.0）。序列、原始下载链接、作者、处理方式和授权见 [CAMERA-SOURCES.md](images/CAMERA-SOURCES.md)。
- 字体：Google Fonts 的 DM Sans 与 Instrument Serif，授权文件在 `fonts/`。
- 点云示意、界面图标与交互为本次制作的 Canvas、SVG 和原生 JavaScript。

阅读清单保存在当前浏览器，换设备或清除浏览器存储不会同步；可随时导出 Markdown 留存。分享筛选视图使用 URL 参数，阅读清单通过导出文件分享。
