# English With Emily - Colab video generation server
# Run this file in Google Colab after your existing Kokoro + SadTalker setup is loaded.
#
# API:
# POST /generate
# JSON: {"script":"Emily: ...\nDavid: ...","mode":"Normal English","captionStyle":"highlight","format":"16:9 YouTube"}
# Returns an MP4 video.

import os, re, json, uuid, subprocess, tempfile, shutil, threading, traceback
from pathlib import Path
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

BASE="/content"
SAD="/content/SadTalker"
SADPY="/content/sadtalker_env/bin/python"
KOKPY="/content/kokoro_env/bin/python"
EMILY_IMG="/content/emily.jpg"
DAVID_IMG="/content/david.jpg"
OUT=Path("/content/english_with_emily_jobs")
OUT.mkdir(exist_ok=True)
JOBS={}

app=Flask(__name__)
CORS(app, resources={r"/*":{"origins":"*"}}, methods=["GET","POST","OPTIONS"], allow_headers=["Content-Type","Authorization"], supports_credentials=False)

@app.after_request
def add_cors_headers(response):
    # Explicit headers keep browser requests working through temporary Colab tunnels.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route("/generate", methods=["OPTIONS"])
def generate_options():
    return ("", 204)

def run(cmd):
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    if p.returncode:
        raise RuntimeError(p.stdout[-5000:])
    return p.stdout

def duration(path):
    s=run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",str(path)])
    return float(s.strip())

def make_audio(text, voice, speed, out):
    script=Path(out).with_suffix(".py")
    script.write_text(f"""
import soundfile as sf
from kokoro import KPipeline
pipe=KPipeline(lang_code='a')
chunks=[]
for _,_,audio in pipe({text!r}, voice={voice!r}, speed={speed}):
    chunks.append(audio)
import numpy as np
audio=np.concatenate(chunks) if chunks else np.zeros(1,dtype=np.float32)
sf.write({str(out)!r}, audio, 24000)
""")
    run([KOKPY,str(script)])
    script.unlink(missing_ok=True)

def make_lipsync(audio,image,outdir):
    Path(outdir).mkdir(parents=True,exist_ok=True)
    run([
        SADPY,f"{SAD}/inference.py",
        "--driven_audio",str(audio),
        "--source_image",str(image),
        "--result_dir",str(outdir),
        "--checkpoint_dir",f"{SAD}/checkpoints",
        "--preprocess","full","--still"
    ])
    vids=sorted(Path(outdir).glob("*.mp4"),key=lambda p:p.stat().st_mtime,reverse=True)
    if not vids:
        raise RuntimeError("SadTalker did not produce an MP4.")
    return vids[0]

def make_segment(active_video, inactive_image, audio, speaker, out):
    # Active speaker occupies one half; the other half remains visible as a clean static portrait.
    filt=(
      "[0:v]scale=672:-2,pad=672:768:(ow-iw)/2:(oh-ih)/2:color=0x111827,"
      "setsar=1[left];"
      "[1:v]scale=672:-2,pad=672:768:(ow-iw)/2:(oh-ih)/2:color=0x111827,"
      "setsar=1[right];"
      "[left][right]hstack=inputs=2,format=yuv420p[v]"
    )
    # Only the static portrait input needs looping. Never apply -loop to the SadTalker MP4.
    if speaker=="Emily":
        cmd=["ffmpeg","-y","-i",str(active_video),"-loop","1","-i",str(inactive_image),"-i",str(audio)]
    else:
        cmd=["ffmpeg","-y","-loop","1","-i",str(inactive_image),"-i",str(active_video),"-i",str(audio)]
    cmd += [
      "-filter_complex",filt,"-map","[v]","-map","2:a",
      "-t",str(duration(audio)),"-r","25","-c:v","libx264","-preset","veryfast",
      "-crf","20","-c:a","aac","-b:a","160k","-shortest",str(out)
    ]
    run(cmd)

def add_captions(video, entries, style, out):
    # Keep the full sentence on screen; highlight the currently spoken word.
    # Word timing is estimated from word lengths so no extra transcription service is required.
    ass=Path(out).with_suffix(".ass")
    styles={
      "highlight": ("Arial",42,"&H00FFFFFF&","&H005EEAD4&"),
      "clean": ("Arial",42,"&H00FFFFFF&","&H00FFFFFF&"),
      "blue": ("Arial",42,"&H00FFFFFF&","&H00FFD166&"),
      "beginner": ("Arial",48,"&H00FFFFFF&","&H0060A5FA&"),
      "bubble": ("Arial",42,"&H00172033&","&H002563EB&"),
    }
    font,size,primary,accent=styles.get(style,styles["highlight"])
    header=f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1344
PlayResY: 768
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},{primary},{primary},&H00000000&,&H99000000&,1,0,0,0,100,100,0,0,1,2,1,2,70,70,65,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    def ts(sec):
        h=int(sec//3600); m=int((sec%3600)//60); s=sec%60
        return f"{h}:{m:02d}:{s:05.2f}"
    lines=[header]
    for e in entries:
        start=e["start"]; end=e["end"]; words=e["text"].split()
        if not words: continue
        total=max(end-start,0.5)
        weights=[max(len(re.sub(r"[^A-Za-z]","",w)),1) for w in words]
        unit=total/sum(weights)
        for i,w in enumerate(words):
            a=start+unit*sum(weights[:i]); b=start+unit*sum(weights[:i+1])
            shown=" ".join(words)
            if style in ("highlight","blue","beginner"):
                # ASS override keeps the entire sentence visible while only the active word changes color.
                before=" ".join(words[:i]); after=" ".join(words[i+1:])
                text=(before+" " if before else "")+"{\\c"+accent+"}"+w+"{\\c"+primary+"}"+(" "+after if after else "")
            else:
                text=shown
            alignment = 1 if e["speaker"]=="Emily" else 3
            # Keep the whole sentence visible for the full sentence duration while the active word changes.
            lines.append(f"Dialogue: 0,{ts(a)},{ts(b)},Default,,0,0,0,,{{\\an{alignment}}}{text}")
    ass.write_text("\n".join(lines),encoding="utf-8")
    run(["ffmpeg","-y","-i",str(video),"-vf",f"ass={ass}","-c:v","libx264","-preset","veryfast","-crf","19","-c:a","copy",str(out)])
    ass.unlink(missing_ok=True)

@app.get("/health")
def health():
    return jsonify({"ok":True,"service":"English With Emily generator"})

def run_job(job_id, script, mode, style, fmt):
    try:
        lines=[]
        for raw in script.splitlines():
            raw=raw.strip()
            m=re.match(r"^(Emily|David)\s*:\s*(.+)$",raw,re.I)
            if m: lines.append((m.group(1).title(),m.group(2).strip()))
        if not lines: raise RuntimeError("Use lines beginning with Emily: or David:")
        job=OUT/job_id; job.mkdir(parents=True,exist_ok=True)
        JOBS[job_id].update(status="processing",progress=0,message="Starting video generation...")
        clips=[]; entries=[]; cursor=0.0
        speed=0.78 if mode=="Slow English" else (0.72 if mode=="Beginner Friendly" else 1.0)
        pause=0.35 if mode=="Normal English" else 0.65
        for n,(speaker,text) in enumerate(lines,1):
            JOBS[job_id].update(progress=int((n-1)/len(lines)*85),message=f"Generating {speaker} segment {n}/{len(lines)}...")
            audio=job/f"audio_{n}.wav"; voice="af_heart" if speaker=="Emily" else "am_michael"
            make_audio(text,voice,speed,audio)
            lipdir=job/f"lip_{n}"
            active=make_lipsync(audio,EMILY_IMG if speaker=="Emily" else DAVID_IMG,lipdir)
            seg=job/f"segment_{n}.mp4"
            make_segment(active,DAVID_IMG if speaker=="Emily" else EMILY_IMG,audio,speaker,seg)
            d=duration(seg); clips.append(seg)
            entries.append({"start":cursor,"end":cursor+d,"text":text,"speaker":speaker})
            cursor+=d+pause
        JOBS[job_id].update(progress=90,message="Joining video and adding captions...")
        concat=job/"concat.txt"; concat.write_text("\n".join(f"file '{str(x)}'" for x in clips))
        joined=job/"joined.mp4"
        run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(concat),"-c:v","libx264","-preset","veryfast","-crf","19","-c:a","aac","-b:a","160k",str(joined)])
        final=job/"English_With_Emily.mp4"; add_captions(joined,entries,style,final)
        JOBS[job_id].update(status="done",progress=100,message="Video ready.",file=str(final))
    except Exception as e:
        JOBS[job_id].update(status="error",progress=0,message=str(e),trace=traceback.format_exc())

@app.post("/generate")
def generate():
    data=request.get_json(force=True)
    script=data.get("script","").strip(); mode=data.get("mode","Normal English"); style=data.get("captionStyle","highlight"); fmt=data.get("format","16:9 YouTube")
    if not script: return jsonify({"error":"Script is empty"}),400
    lines=[raw for raw in script.splitlines() if re.match(r"^\s*(Emily|David)\s*:\s*.+$",raw,re.I)]
    if not lines: return jsonify({"error":"Use lines beginning with Emily: or David:"}),400
    job_id=uuid.uuid4().hex
    JOBS[job_id]={"status":"queued","progress":0,"message":"Queued..."}
    threading.Thread(target=run_job,args=(job_id,script,mode,style,fmt),daemon=True).start()
    return jsonify({"job_id":job_id,"status_url":f"/status/{job_id}"}),202

@app.get("/status/<job_id>")
def job_status(job_id):
    job=JOBS.get(job_id)
    if not job: return jsonify({"error":"Job not found"}),404
    out={k:v for k,v in job.items() if k!="trace"}
    if job.get("status")=="done": out["download_url"]=f"/download/{job_id}"
    return jsonify(out)

@app.get("/download/<job_id>")
def download_job(job_id):
    job=JOBS.get(job_id)
    if not job or job.get("status")!="done": return jsonify({"error":"Video not ready"}),404
    return send_file(job["file"],mimetype="video/mp4",as_attachment=True,download_name="English_With_Emily.mp4")



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
