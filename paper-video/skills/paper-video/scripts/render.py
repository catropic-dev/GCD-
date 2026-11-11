import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import wave

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageOps


def require_text(data, key):
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be nonempty text")
    return value


def require_list(data, key):
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a nonempty array")
    return value


def require_lines(data, key):
    lines = require_list(data, key)
    if any(not isinstance(line, str) or not line.strip() or "\n" in line or "\r" in line for line in lines):
        raise ValueError(f"{key} must contain nonempty single-line strings")
    return lines


def validate(story):
    require_text(story, "title")
    for chapter in require_list(story, "chapters"):
        for key in ("section", "title", "reference"):
            require_text(chapter, key)
        if len(require_lines(chapter, "points")) > 3:
            raise ValueError("Use at most three points per chapter")
        narration = require_lines(chapter, "narration")
        if "spoken" in chapter and len(require_lines(chapter, "spoken")) != len(narration):
            raise ValueError("spoken and narration must have the same number of segments")
        for entry in require_list(chapter, "qa"):
            require_text(entry, "question")
            require_text(entry, "answer")
            require_lines(entry, "keywords")


def draw_text(draw, text, xy, size, minimum, width, rows, fill, font_path):
    for scale in range(size, minimum - 1, -1):
        font = ImageFont.truetype(str(font_path), scale)
        lines = [""]
        for char in text:
            if char == "\n":
                lines.append("")
                continue
            if lines[-1] and draw.textlength(lines[-1] + char, font=font) > width:
                lines.append("")
            lines[-1] += char
        if len(lines) <= rows:
            for index, line in enumerate(lines):
                draw.text((xy[0], xy[1] + index * (scale + 5)), line, font=font, fill=fill)
            return
    raise ValueError(f"Shorten this text to fit the slide: {text[:100]}")


def slide(story, chapter, step, asset_dir, font):
    background = "#101b2c"
    white, muted, accent = "#f4f7fb", "#a8bed5", "#73e5c5"
    image = Image.new("RGB", (1280, 720), background)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1279, 7), fill=accent)
    draw_text(draw, story["title"], (60, 37), 23, 16, 1160, 1, accent, font)
    draw_text(draw, chapter["title"], (60, 81), 42, 25, 1160, 2, white, font)
    draw_text(draw, chapter["reference"], (60, 182), 21, 15, 1160, 1, muted, font)
    draw.rounded_rectangle((60, 230, 1220, 554), radius=21, fill="#192a40")
    illustration = chapter.get("image")
    if illustration:
        with Image.open(asset_dir / illustration) as original:
            figure = ImageOps.contain(original.convert("RGB"), (535, 288))
        image.paste(figure, (660 + (535 - figure.width) // 2, 248 + (288 - figure.height) // 2))
    points = chapter["points"]
    for index, point in enumerate(points):
        y = 256 + index * 96
        color = accent if index == min(step, len(points) - 1) else white
        draw.ellipse((87, y + 14, 97, y + 24), fill=color)
        draw_text(draw, point, (115, y), 31, 21, 495 if illustration else 1050, 2, color, font)
    draw.rounded_rectangle((60, 578, 1220, 692), radius=16, fill="#091321")
    draw_text(draw, chapter["narration"][step], (88, 591), 23, 17, 1100, 3, white, font)
    return image


def stamp(seconds):
    ms = round(seconds * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02}.{ms % 1000:03}"


def audio_jobs(chapters, audio_dir):
    return [
        {"path": str(audio_dir / f"voice-{index:02}-{step:02}.wav"), "text": text}
        for index, chapter in enumerate(chapters)
        for step, text in enumerate(chapter.get("spoken", chapter["narration"]))
    ]


async def neural_narration(jobs, voice, ffmpeg):
    try:
        import edge_tts
    except ImportError as error:
        raise ValueError("Install edge-tts to use --neural-voice") from error
    for job in jobs:
        wav = Path(job["path"])
        mp3 = wav.with_suffix(".mp3")
        try:
            speech = edge_tts.Communicate(job["text"], voice, rate="-6%", connect_timeout=10, receive_timeout=30)
            await asyncio.wait_for(speech.save(str(mp3)), timeout=60)
        except Exception as error:
            raise ValueError(f"Online speech failed for {wav.stem}: {error}") from error
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(mp3),
                        "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(wav)],
                       check=True, timeout=30)


def render(story, asset_dir, output, font, supplied_audio, ffmpeg, neural_voice):
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        work = Path(temporary)
        stage = work / "delivery"
        stage.mkdir()
        audio_dir = supplied_audio or work
        jobs = audio_jobs(story["chapters"], audio_dir)
        if neural_voice:
            asyncio.run(neural_narration(jobs, neural_voice, ffmpeg))
        elif supplied_audio is None:
            if os.name != "nt":
                raise ValueError("Provide --audio-dir or --neural-voice outside Windows")
            job_file = work / "speech.json"
            job_file.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
            script = Path(__file__).with_name("narrate.ps1")
            subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                            "-File", str(script), str(job_file)], check=True, timeout=1800)

        frames = ["ffconcat version 1.0"]
        captions = ["WEBVTT"]
        samples = 0
        params = None
        clips = iter(jobs)
        with wave.open(str(work / "narration.wav"), "wb") as audio:
            for index, chapter in enumerate(story["chapters"]):
                for step, text in enumerate(chapter["narration"]):
                    with wave.open(next(clips)["path"], "rb") as clip:
                        fmt = (clip.getnchannels(), clip.getsampwidth(), clip.getframerate())
                        if params is None:
                            params = fmt
                            audio.setnchannels(fmt[0])
                            audio.setsampwidth(fmt[1])
                            audio.setframerate(fmt[2])
                        if fmt != params or not clip.getnframes():
                            raise ValueError("All WAV clips must be nonempty and share one PCM format")
                        start = samples / fmt[2]
                        if step == 0:
                            chapter["start"] = start
                        padding = round(0.38 * fmt[2])
                        silence = (b"\x80" if fmt[1] == 1 else bytes(fmt[1])) * padding * fmt[0]
                        audio.writeframesraw(clip.readframes(clip.getnframes()) + silence)
                        count = clip.getnframes() + padding
                        samples += count
                        end = samples / fmt[2]
                    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    captions.append(f"{stamp(start)} --> {stamp(end)}\n{escaped}")
                    frame = f"frame-{index:03}-{step:03}.png"
                    slide(story, chapter, step, asset_dir, font).save(work / frame)
                    frames.extend([f"file '{frame}'", "option framerate 1000", f"duration {count / fmt[2]:.9f}"])
        frames.extend([f"file '{frame}'", "option framerate 1000"])
        (work / "frames.ffconcat").write_text("\n".join(frames) + "\n", encoding="ascii")
        (stage / "captions.vtt").write_text("\n\n".join(captions) + "\n", encoding="utf-8")

        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", "frames.ffconcat", "-i", "narration.wav",
            "-vf", "fps=12", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
            "-t", f"{end:.6f}", str(stage / "lecture.mp4"),
        ], cwd=work, check=True)
        viewer = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
        data = json.dumps(story, ensure_ascii=False).replace("<", "\\u003c")
        (stage / "index.html").write_text(viewer.replace("/* STORY_DATA */", data), encoding="utf-8")
        output.mkdir()
        for artifact in stage.iterdir():
            shutil.move(artifact, output / artifact.name)
    print(f"{len(story['chapters'])} chapters, {end:.1f} seconds: {output}")


def main():
    parser = argparse.ArgumentParser(description="Render a narrated paper walkthrough and interactive player.")
    parser.add_argument("storyboard", type=Path)
    parser.add_argument("output", type=Path, help="New output directory")
    audio = parser.add_mutually_exclusive_group()
    audio.add_argument("--audio-dir", type=Path)
    audio.add_argument("--neural-voice", nargs="?", const="zh-CN-XiaoxiaoNeural",
                       help="Send spoken narration text to Microsoft's online speech service")
    parser.add_argument("--ffmpeg", type=Path, help="FFmpeg executable with libx264 support")
    parser.add_argument("--font", type=Path, default=Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msyh.ttc")
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        if output.exists():
            raise ValueError("Choose a new output directory")
        story = json.loads(args.storyboard.read_text(encoding="utf-8-sig"))
        validate(story)
        output.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = str(args.ffmpeg.resolve()) if args.ffmpeg else imageio_ffmpeg.get_ffmpeg_exe()
        render(story, args.storyboard.resolve().parent, output, args.font.resolve(),
               args.audio_dir.resolve() if args.audio_dir else None, ffmpeg, args.neural_voice)
    except (OSError, ValueError, AttributeError, TypeError, wave.Error, subprocess.SubprocessError) as error:
        parser.exit(1, f"Render failed: {error}\n")


if __name__ == "__main__":
    main()
