# 论文视频讲解

这是一个 Codex 插件。把论文、自己的阅读笔记和讲解顺序交给 Codex，技能会编排中文旁白、画面和问答，再调用渲染器输出 MP4、WebVTT 和互动播放器。

在 Codex 桌面端打开本仓库，重启应用后，在「插件」中切换到「GCD 研究工具」，安装「论文视频讲解」。也可以在终端运行 `codex plugin marketplace add catropic-dev/GCD-` 添加仓库来源，再在桌面端安装。

新建聊天，从技能选择器选择 `paper-video`，附上一份论文 PDF，然后说「把附件生成中文论文讲解视频」。Codex 会从论文编排分镜。你也可以附笔记，或指定先讲哪个例子、图或公式。

## 本地渲染

需要 Python 3.10+ 和中文字体。Windows 默认使用微软雅黑和已安装的简体中文 SAPI 语音。已有配音时可传入逐段 WAV 文件，也可在其他平台渲染。

```powershell
python -m pip install -r paper-video/requirements.txt
python paper-video/skills/paper-video/scripts/render.py storyboard.json output/paper-demo
```

输出目录应为新目录。自定义选项见 `--help`。手工提供 WAV 时使用 `--audio-dir`，文件按从零开始的章节和旁白序号命名，如 `voice-00-00.wav`。所有 WAV 须使用相同 PCM 格式。非 Windows 环境通过 `--font` 指定中文字体。

分镜 JSON 示例。每段旁白对应一次画面切换和一条字幕。每章最多三条屏幕要点，`image` 是可选图片，路径相对于分镜文件。

```json
{
  "title": "广义类别发现",
  "chapters": [
    {
      "section": "任务设定",
      "title": "未标注数据里有哪些类？",
      "reference": "论文 PDF 第 2 页 §3.1",
      "points": ["已标注：部分猫和狗", "未标注：猫、狗、狐狸和狼"],
      "narration": ["训练时，只能看到一部分猫和狗的标签。", "剩下的猫狗，与所有狐狸和狼，一起进入未标注的数据池。"],
      "qa": [
        {
          "question": "未标注数据只有新类吗？",
          "answer": "未标注池同时包含旧类与新类。部分旧类样本也没有标签。",
          "keywords": ["新类", "旧类", "未标注"]
        }
      ]
    }
  ]
}
```

打开 `index.html` 可播放、变速、查看讲稿、提问并跳到对应时间。窗口在本地匹配分镜中预写的问答，匹配不到时提示改用术语或将问题发给 Codex。页面可离线打开，字幕已经写入视频画面，并额外提供 `captions.vtt`。
