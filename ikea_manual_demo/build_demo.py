# Self-contained interactive 3D assembly manual from REAL IKEA meshes (Bench/sjalland).
# Parts START laid flat on the floor (real setup), then rise + rotate into place, one per step.
import json, os, math

ROOT="/Users/maksym/Documents/Files/Heavy/Work/research_waterloo/IKEA-Manuals-at-Work"
PARTDIR=os.path.join(ROOT,"data/parts/Bench/sjalland")
OUT=os.path.join(os.path.dirname(__file__),"sjalland_manual.html")

def load_part(path):
    V=[];F=[]
    for ln in open(path):
        if ln.startswith("v "):
            p=ln.split(); V.append((float(p[1]),float(p[3]),float(p[2])))   # (x,z,y) -> z up
        elif ln.startswith("f "):
            idx=[int(t.split("/")[0])-1 for t in ln.split()[1:]]
            for m in range(1,len(idx)-1): F.append((idx[0],idx[m],idx[m+1]))
    return V,F
files=sorted(f for f in os.listdir(PARTDIR) if f.endswith(".obj"))
parts=[load_part(os.path.join(PARTDIR,f)) for f in files]
def cent(V): n=len(V);return (sum(p[0] for p in V)/n,sum(p[1] for p in V)/n,sum(p[2] for p in V)/n)
def rotx(p,a,c): y=p[1]-c[1];z=p[2]-c[2];C=math.cos(a);S=math.sin(a);return (p[0],c[1]+y*C-z*S,c[2]+y*S+z*C)
def rotz(p,a,c): x=p[0]-c[0];y=p[1]-c[1];C=math.cos(a);S=math.sin(a);return (c[0]+x*C-y*S,c[1]+x*S+y*C,p[2])
def roty(p,a,c): x=p[0]-c[0];z=p[2]-c[2];C=math.cos(a);S=math.sin(a);return (c[0]+x*C+z*S,p[1],c[2]-x*S+z*C)

FLOOR_Z=-0.36
# floor layout per part: (rotx_deg, roty_deg, rotz_deg yaw, scatter_x, scatter_y, lift)
# loose pile dumped from the box: ALL parts lie flat on the floor, overlapping a little.
# leg-frames are thin panels -> roty +/-90 lays the panel face-down flat (rotx would stand them on edge)
FLOOR_POSE={0:(0,0,4,-0.05,-0.25,0.00),     # seat slat: already flat
            1:(0,-90,12,-0.62,0.18,0.06),   # left leg-frame: roty -90 lays it flat
            2:(0,0,52,0.15,0.18,0.11),      # stretcher bar: flat, lying over the pile
            3:(0,90,-12,0.66,0.32,0.00)}    # right leg-frame: roty +90 lays it flat
NAME={0:"top seat slat",1:"left side frame",2:"bottom stretcher bar",3:"right side frame"}
WOOD={0:"#bb8a52",1:"#9c6b3f",2:"#82562d",3:"#9c6b3f"}; ACTIVE="#e8902a"
ORDER=[1,3,2,0]
STEP_TXT=[
 "Step 1 / 4 - Stand the LEFT side frame up off the floor.",
 "Step 2 / 4 - Stand the RIGHT side frame up.",
 "Step 3 / 4 - Lift the bottom STRETCHER bar between the side frames.",
 "Step 4 / 4 - Lay the top SEAT slat across the frames.  ->  bench assembled.",
]
LIGHT=dict(ambient=0.55,diffuse=0.85,specular=0.22,roughness=0.65,fresnel=0.12)
LPOS=dict(x=120,y=-200,z=260)
CENT=[cent(V) for V,_ in parts]

# precompute floor translation T[pi] (at full rotation) so part rests on floor at scatter spot
FLOOR_T={}
for pi,(V,F) in enumerate(parts):
    rx,ry,rz,sx,sy,lift=FLOOR_POSE[pi]; c=CENT[pi]
    Vr=[rotz(roty(rotx(v,math.radians(rx),c),math.radians(ry),c),math.radians(rz),c) for v in V]
    cr=cent(Vr); mnz=min(v[2] for v in Vr)
    FLOOR_T[pi]=(sx-cr[0], sy-cr[1], (FLOOR_Z+lift)-mnz)

def pose_at(pi,u):
    # u=0 -> flat on floor, u=1 -> assembled
    V,F=parts[pi]; rx,ry,rz,sx,sy,lift=FLOOR_POSE[pi]; c=CENT[pi]; T=FLOOR_T[pi]; k=1.0-u
    ax=math.radians(rx)*k; ay=math.radians(ry)*k; az=math.radians(rz)*k
    out=[]
    for v in V:
        w=rotz(roty(rotx(v,ax,c),ay,c),az,c)
        out.append((w[0]+T[0]*k, w[1]+T[1]*k, w[2]+T[2]*k))
    return out

def state(part,step):
    pos=ORDER.index(part); return "placed" if pos<step else ("active" if pos==step else "waiting")

def part_trace(pi,step,t):
    st=state(pi,step); u = 1.0 if st=="placed" else (t if st=="active" else 0.0)
    V=pose_at(pi,u); F=parts[pi][1]
    color = ACTIVE if st=="active" else WOOD[pi]
    return {"type":"mesh3d","x":[v[0] for v in V],"y":[v[1] for v in V],"z":[v[2] for v in V],
            "i":[f[0] for f in F],"j":[f[1] for f in F],"k":[f[2] for f in F],
            "color":color,"opacity":1.0,"flatshading":False,"lighting":LIGHT,
            "lightposition":LPOS,"hoverinfo":"skip","showscale":False}

def floor_trace():
    z=FLOOR_Z-0.004
    return {"type":"mesh3d","x":[-2.7,2.7,2.7,-2.7],"y":[-2.1,-2.1,2.7,2.7],"z":[z,z,z,z],
            "i":[0,0],"j":[1,2],"k":[2,3],"color":"#d8cfc1","opacity":1.0,"flatshading":True,
            "lighting":dict(ambient=0.7,diffuse=0.6,specular=0.05),"lightposition":LPOS,
            "hoverinfo":"skip","showscale":False}

def labels(step,t):
    act=ORDER[step]; u=t; V=pose_at(act,u)
    n=len(V); cx=sum(v[0] for v in V)/n; cy=sum(v[1] for v in V)/n; cz=sum(v[2] for v in V)/n
    h=CENT[act]
    return [
      {"type":"scatter3d","x":[h[0]],"y":[h[1]],"z":[h[2]],"mode":"markers+text",
       "marker":{"size":6,"color":"#e23b3b"},"text":["HERE"],"textposition":"top center",
       "textfont":{"color":"#e23b3b","size":12},"hoverinfo":"skip","showlegend":False},
      {"type":"scatter3d","x":[cx],"y":[cy],"z":[cz+0.12],"mode":"text",
       "text":["TAKE"],"textfont":{"color":"#ff8c1a","size":13},"hoverinfo":"skip","showlegend":False},
    ]

def frame_data(step,t):
    return [floor_trace()] + [part_trace(p,step,t) for p in range(4)] + labels(step,t)

PER=10; frames=[]; slider=[]; k=0
for step in range(4):
    for f in range(PER+1):
        nm="f%d"%k
        frames.append({"name":nm,"data":frame_data(step,f/PER),"layout":{"title":{"text":STEP_TXT[step]}}})
        slider.append({"label":"","method":"animate","args":[[nm],{"mode":"immediate","frame":{"duration":0,"redraw":True}}]})
        k+=1

noax={"showbackground":False,"showgrid":False,"zeroline":False,"showticklabels":False,"title":""}
layout={"title":{"text":STEP_TXT[0]},"paper_bgcolor":"#eef0f2","width":1000,"height":720,
 "scene":{"xaxis":noax,"yaxis":noax,"zaxis":noax,"aspectmode":"data","bgcolor":"#eef0f2",
          "camera":{"eye":{"x":1.6,"y":-1.75,"z":0.85}}},
 "updatemenus":[{"type":"buttons","showactive":False,"x":0,"y":1,
    "buttons":[{"label":"Play","method":"animate","args":[None,{"frame":{"duration":70,"redraw":True},"fromcurrent":True,"transition":{"duration":0}}]},
               {"label":"Pause","method":"animate","args":[[None],{"mode":"immediate","frame":{"duration":0,"redraw":False}}]}]}],
 "sliders":[{"active":0,"x":0.08,"len":0.86,"currentvalue":{"visible":False},"steps":slider}]}
data0=frame_data(0,0.0)
html=("<!doctype html><html><head><meta charset='utf-8'>"
 "<script src='https://cdn.plot.ly/plotly-2.35.0.min.js'></script>"
 "<style>body{margin:0;font-family:Helvetica,Arial,sans-serif;background:#eef0f2}"
 "h2{margin:10px 16px 0}p{margin:2px 16px 10px;color:#5b6472}</style></head><body>"
 "<h2>IKEA Sjalland bench - assembly manual (real meshes, parts start on the floor)</h2>"
 "<p>Press Play. The real parts lie flat on the floor, then rise into place. Orange = part moving now, "
 "red HERE = where it goes. Drag to rotate, scroll to zoom.</p>"
 "<div id='g'></div><script>"
 "var data="+json.dumps(data0)+";var layout="+json.dumps(layout)+";var frames="+json.dumps(frames)+";"
 "Plotly.newPlot('g',data,layout,{responsive:true}).then(function(){Plotly.addFrames('g',frames);});"
 "</script></body></html>")
open(OUT,"w").write(html)

# validation
for fr in frames:
    for tr in fr["data"]:
        if tr["type"]=="mesh3d" and tr["i"]:
            n=len(tr["x"]); assert max(max(tr["i"]),max(tr["j"]),max(tr["k"]))<n
for pi in range(4):
    Vf=pose_at(pi,0.0); mnz=min(v[2] for v in Vf); rx,ry,rz,sx,sy,lift=FLOOR_POSE[pi]
    cr=cent(Vf); assert abs(mnz-(FLOOR_Z+lift))<1e-6, (pi,mnz); assert abs(cr[0]-sx)<1e-6 and abs(cr[1]-sy)<1e-6
    Va=pose_at(pi,1.0); assert all(abs(Va[k][d]-parts[pi][0][k][d])<1e-9 for k in range(0,len(Va),50) for d in range(3))
print("OK wrote",OUT,"| frames",len(frames),"| size %.2f MB"%(os.path.getsize(OUT)/1e6))
print("floor start verified (parts on floor at z=%.2f); assembled end == real coords"%FLOOR_Z)
