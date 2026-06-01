import os
import subprocess
import uuid
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance


st.set_page_config(
    page_title="PrivAIMin Attendance",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "PrivAIMin"
APP_SUBTITLE = "Private AI Minutes"

DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_COLLECTION = "tryface"
ATTENDANCE_COLLECTION = "attendance_events"

IMAGE_SIZE = 160
MIN_FACE_SIZE = 60
PROB_THRESHOLD = 0.70

DEFAULT_THRESHOLD = 0.60
DEFAULT_GAP = 0.05
DEFAULT_TOP_K = 5

EXTRA_BOX_SCALE = 0.15
FORCE_WHITE_BG = True
FEATHER = 3

ATTENDANCE_CSV = "attendance_log.csv"
CAPTURE_DIR = Path("attendance_captures")
CAPTURE_DIR.mkdir(exist_ok=True)

MY_DATASET_DIR = Path("my_dataset")
CLEAN_FACE_DIR = Path("clean_faces_4_white")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "manual_queue" not in st.session_state:
    st.session_state.manual_queue = []

if "threshold" not in st.session_state:
    st.session_state.threshold = DEFAULT_THRESHOLD

if "gap" not in st.session_state:
    st.session_state.gap = DEFAULT_GAP

if "top_k" not in st.session_state:
    st.session_state.top_k = DEFAULT_TOP_K

if "qdrant_host" not in st.session_state:
    st.session_state.qdrant_host = DEFAULT_QDRANT_HOST

if "qdrant_port" not in st.session_state:
    st.session_state.qdrant_port = DEFAULT_QDRANT_PORT

if "collection" not in st.session_state:
    st.session_state.collection = DEFAULT_COLLECTION

if "dashboard_view" not in st.session_state:
    st.session_state.dashboard_view = "recent"

if "dashboard_search" not in st.session_state:
    st.session_state.dashboard_search = ""

if "dashboard_day" not in st.session_state:
    st.session_state.dashboard_day = None

if "analytics_day" not in st.session_state:
    st.session_state.analytics_day = None


st.markdown("""
<style>
:root{
    --ink:#051F20;
    --deep:#0B2B26;
    --forest:#163832;
    --green:#235347;
    --sage:#8EB69B;
    --mint:#DAF1DE;
    --page:#F6FAF7;
    --card:#FFFFFF;
    --muted:#7B8A82;
    --line:rgba(5,31,32,.08);
    --shadow:0 14px 35px rgba(5,31,32,.08);
    --soft-shadow:0 8px 20px rgba(5,31,32,.06);
    --radius:22px;
}
html, body, [data-testid="stAppViewContainer"]{
    background:#FFFFFF !important;
    color:var(--ink) !important;
    font-family:Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
[data-testid="stHeader"]{background:transparent !important;}
.block-container{
    max-width:1320px !important;
    padding:1.45rem 1.7rem 2.5rem 1.7rem !important;
}
section.main > div.block-container{
    background:#FFFFFF !important;
    border-radius:32px;
    box-shadow:0 28px 65px rgba(5,31,32,.12);
    border:1px solid rgba(255,255,255,.85);
}

section[data-testid="stSidebar"]{
    background:#F7FBF8 !important;
    border-right:1px solid rgba(5,31,32,.06) !important;
    width:245px !important;
    box-shadow:12px 0 34px rgba(5,31,32,.05);
}
section[data-testid="stSidebar"] > div{padding:1.75rem 1.35rem !important;}
.brand-box{display:flex;align-items:center;gap:11px;margin:28px 0 34px 0;padding:0 6px;border-bottom:none !important;}
.brand-icon{width:34px;height:34px;border-radius:50%;background:var(--green);color:white;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:14px;box-shadow:0 10px 22px rgba(35,83,71,.22);}
.brand-title{font-size:14px;font-weight:900;color:var(--deep);letter-spacing:-.03em;}
.brand-subtitle{font-size:9px;color:var(--muted);margin-top:3px;}
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4{
    color:#8C9991 !important;
    font-size:9px !important;
    font-weight:900 !important;
    text-transform:uppercase;
    letter-spacing:.17em;
    margin:24px 0 8px 9px !important;
}
section[data-testid="stSidebar"] div.stButton > button{
    position:relative;
    background:transparent !important;
    color:#718078 !important;
    border:none !important;
    border-radius:13px !important;
    height:39px !important;
    justify-content:flex-start !important;
    text-align:left !important;
    padding-left:18px !important;
    font-size:12px !important;
    font-weight:650 !important;
    box-shadow:none !important;
    margin-bottom:4px !important;
    transition:all .18s ease !important;
}
section[data-testid="stSidebar"] div.stButton > button:hover{
    background:#E6F5EA !important;
    color:var(--green) !important;
    transform:translateX(3px);
}
section[data-testid="stSidebar"] div.stButton > button:focus{
    background:#E1F2E6 !important;
    color:var(--deep) !important;
    font-weight:850 !important;
}
section[data-testid="stSidebar"] div.stButton > button:focus:before{
    content:"";position:absolute;left:0;top:8px;width:4px;height:23px;border-radius:999px;background:var(--green);
}
section[data-testid="stSidebar"] div.stButton > button *{color:inherit !important;}
section[data-testid="stSidebar"] .stExpander{display:none !important;}
section[data-testid="stSidebar"] hr{display:none !important;}

.topbar{display:flex;align-items:center;justify-content:space-between;gap:18px;margin-bottom:18px;}
.search-pill{height:42px;min-width:330px;background:#F8FBF9;border:1px solid var(--line);border-radius:18px;display:flex;align-items:center;padding:0 16px;color:#9AA69F;font-size:12px;box-shadow:var(--soft-shadow);}
.top-user{display:flex;align-items:center;gap:10px;background:#F8FBF9;border:1px solid var(--line);border-radius:18px;padding:7px 13px;box-shadow:var(--soft-shadow);}
.user-avatar{width:34px;height:34px;border-radius:50%;background:#DAF1DE;display:flex;align-items:center;justify-content:center;color:#235347;font-size:12px;font-weight:900;}
.user-name{font-size:12px;font-weight:900;color:var(--deep);line-height:1.1;}.user-mail{font-size:10px;color:var(--muted);}
.title-row{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:16px;}
.title-main{font-size:31px;font-weight:950;color:var(--ink);letter-spacing:-.05em;margin:0;}.title-sub{font-size:13px;color:var(--muted);margin-top:4px;}
.header-buttons{display:flex;gap:10px;align-items:center;}.mini-btn{border-radius:999px;padding:12px 18px;font-size:12px;font-weight:850;border:1px solid var(--line);background:#fff;color:var(--deep);box-shadow:var(--soft-shadow);}.mini-btn.primary{background:var(--green);color:#fff;border-color:var(--green);}

.page-top,.page-header{display:none !important;}
h1,h2,h3{color:var(--ink) !important;letter-spacing:-.035em;}h2{font-size:23px !important;}h3{font-size:17px !important;}p,label,.stCaption{color:var(--muted) !important;}

.card,.stat-card,.panel-card,.analytics-card,.analytics-wrap,.dash-card{background:#fff;border:1px solid var(--line);border-radius:var(--radius);padding:20px;box-shadow:var(--soft-shadow);}
.stat-card{min-height:126px;position:relative;overflow:hidden;transition:all .18s ease;}.stat-card:hover,.dash-card:hover{transform:translateY(-2px);box-shadow:var(--shadow);}
.stat-label{color:#4D5A54;font-size:13px;font-weight:850;margin-bottom:14px;}.stat-value{color:var(--ink);font-size:38px;font-weight:950;line-height:1;letter-spacing:-.05em;}.stat-hint{display:flex;align-items:center;gap:6px;color:#7FA68A;font-size:11px;font-weight:800;margin-top:12px;}.stat-card.active{background:var(--green);}.stat-card.active .stat-label,.stat-card.active .stat-value{color:white;}.stat-card.active .stat-hint{color:#CDEBD4;}
.stat-corner{width:31px;height:31px;min-width:31px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#F3F8F5;border:1px solid var(--line);color:var(--green) !important;font-weight:900;text-decoration:none !important;transition:all .18s ease;}.stat-corner:hover{background:#DAF1DE;transform:scale(1.06);}.stat-card.active .stat-corner{background:white;color:var(--green) !important;}
.card-action-note{font-size:11px;color:#7B8A82;font-weight:800;margin-top:6px;text-align:center;}
.dashboard-detail-title{font-size:18px;font-weight:950;color:var(--ink);margin-bottom:4px;}
.dashboard-detail-sub{font-size:12px;color:var(--muted);font-weight:750;margin-bottom:12px;}

.progress-line{display:none;}

.dashboard-grid{display:grid;grid-template-columns:1.25fr .72fr .72fr;gap:14px;margin-top:16px;}.dashboard-grid-2{display:grid;grid-template-columns:1.05fr 1fr .72fr;gap:14px;margin-top:14px;}.dash-title{font-size:15px;font-weight:900;color:var(--ink);margin-bottom:14px;}.bar-week{display:flex;align-items:flex-end;justify-content:space-between;height:146px;gap:13px;margin-top:6px;}.bar-wrap{flex:1;text-align:center;}.bar{height:118px;border-radius:999px;background:repeating-linear-gradient(135deg,#E2EAE5 0 4px,#fff 4px 8px);position:relative;overflow:hidden;}.bar-fill{position:absolute;bottom:0;left:0;width:100%;background:var(--green);border-radius:999px;}.bar-fill.light{background:var(--sage);}.bar-label{font-size:11px;color:#8B9791;margin-top:8px;font-weight:800;}.bar-count{font-size:10px;color:#235347;margin-top:3px;font-weight:900;} .bar-help{font-size:11px;color:#7B8A82;font-weight:750;margin-top:22px;margin-bottom:10px;}.list-row{display:flex;align-items:center;justify-content:space-between;padding:9px 0;border-bottom:1px solid rgba(5,31,32,.06);}.list-row:last-child{border-bottom:0;}.avatar-dot{width:33px;height:33px;border-radius:50%;background:#DAF1DE;display:flex;align-items:center;justify-content:center;color:var(--green);font-weight:900;font-size:11px;margin-right:10px;}.row-left{display:flex;align-items:center;min-width:0;}.row-name{font-size:12px;font-weight:850;color:var(--deep);}.row-sub{font-size:10px;color:var(--muted);}.tag{border-radius:999px;padding:4px 8px;font-size:9px;font-weight:850;}.tag.ok{background:#E3F4E7;color:var(--green);}.tag.warn{background:#E5F5E9;color:#235347;}.tag.bad{background:#E5F5E9;color:#235347;}.gauge-card{text-align:center;}.gauge{width:178px;height:92px;margin:4px auto 4px auto;border-radius:178px 178px 0 0;background:conic-gradient(from 270deg,var(--green) 0deg,var(--green) calc(var(--gauge-deg, 72) * 1.8deg),#E6EFE9 calc(var(--gauge-deg, 72) * 1.8deg),#E6EFE9 180deg);position:relative;overflow:hidden;}.gauge:after{content:"";position:absolute;left:28px;right:28px;bottom:-58px;height:116px;border-radius:50%;background:white;}.gauge-value{font-size:38px;font-weight:950;color:var(--ink);margin-top:-43px;position:relative;z-index:2;}.legend{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:12px;font-size:11px;color:var(--muted);}.legend span:before{content:"";display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--green);margin-right:6px;}.legend span:nth-child(2):before{background:var(--sage)}.legend span:nth-child(3):before{background:#DAF1DE}.time-card{background:radial-gradient(circle at 80% 0%,#235347,#051F20 70%);color:white;border-radius:22px;min-height:168px;display:flex;flex-direction:column;justify-content:center;align-items:center;box-shadow:var(--shadow);}.time-card .time{font-size:30px;font-weight:950;letter-spacing:-.03em;}.time-card .label{color:#CDEBD4;font-size:13px;font-weight:850;margin-bottom:8px;}

div.stButton > button,.stDownloadButton button{background:var(--green) !important;color:white !important;border:none !important;border-radius:15px !important;height:42px !important;font-weight:850 !important;box-shadow:none !important;transition:all .16s ease !important;}div.stButton > button:hover,.stDownloadButton button:hover{background:var(--deep) !important;transform:translateY(-1px);}div.stButton > button *{color:inherit !important;}
input,textarea,select,[data-baseweb="input"],[data-baseweb="select"]{border-radius:14px !important;}div[data-testid="stDataFrame"]{border-radius:18px;overflow:hidden;border:1px solid var(--line);box-shadow:var(--soft-shadow);}.stAlert{border-radius:16px !important;border:none !important;}.badge{display:inline-block;padding:6px 12px;border-radius:999px;font-size:12px;font-weight:850;}.badge.present{background:#E5F5E9;color:var(--green);}.badge.left{background:#E5F5E9;color:#235347;}.badge.late{background:#E5F5E9;color:#235347;}.badge.absent{background:#EFF3F0;color:#6F7D75;}

.analytics-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:14px 0;}.analytics-card{border-radius:22px;padding:20px;}.analytics-card:first-child{background:var(--green);}.analytics-card:first-child .analytics-label,.analytics-card:first-child .analytics-value{color:white !important;}.analytics-label{font-size:12px;color:var(--muted);font-weight:850;margin-bottom:10px;}.analytics-value{font-size:34px;font-weight:950;color:var(--ink);letter-spacing:-.04em;}.analytics-wrap{border-radius:22px;margin-top:14px;}.analytics-row{display:flex;align-items:center;gap:12px;padding:12px 0;border-bottom:1px solid rgba(5,31,32,.06);}.analytics-row:last-child{border-bottom:0;}.analytics-status{min-width:95px;font-size:13px;font-weight:900;color:var(--deep);}.analytics-count{min-width:48px;text-align:right;font-size:12px;color:var(--muted);font-weight:850;}.analytics-bar-bg{flex:1;height:22px;background:#EEF4F0;border-radius:999px;overflow:hidden;}.analytics-bar-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--sage),var(--green));}.percent-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px;}.percent-card{background:white;border:1px solid var(--line);border-radius:22px;padding:20px;box-shadow:var(--soft-shadow);text-align:center;}
.camera-preview{height:240px;background:radial-gradient(circle at 50% 20%,#235347,#051F20 72%);border-radius:22px;display:flex;align-items:center;justify-content:center;color:#DAF1DE;flex-direction:column;}
div[data-testid="stTextInput"] input{height:42px;border-radius:18px !important;background:#F8FBF9 !important;border:1px solid var(--line) !important;box-shadow:var(--soft-shadow) !important;font-size:12px !important;}
@media(max-width:1000px){.block-container{padding:1rem !important;}section.main > div.block-container{border-radius:22px;}.dashboard-grid,.dashboard-grid-2,.analytics-grid,.percent-grid{grid-template-columns:1fr;}.topbar,.title-row{flex-direction:column;align-items:flex-start;}.search-pill{min-width:100%;}}


.mini-date-real{background:white;border:1px solid rgba(5,31,32,.08);border-radius:999px;padding:13px 15px;text-align:center;font-size:12px;font-weight:850;color:var(--deep);box-shadow:var(--soft-shadow);height:44px;display:flex;align-items:center;justify-content:center;}
.clean-dashboard{grid-template-columns:1fr .85fr .9fr !important;}
div[data-testid="stTextInput"] input{height:42px;border-radius:18px !important;background:#F8FBF9 !important;border:1px solid var(--line) !important;box-shadow:var(--soft-shadow) !important;font-size:12px !important;}
@media(max-width:1000px){.clean-dashboard{grid-template-columns:1fr !important;}.mini-date-real{width:100%;}}

.stat-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;}


.analytics-extra-grid{display:grid;grid-template-columns:.8fr 1.4fr;gap:14px;margin-top:14px;}
.growth-chart-card,.day-share-card{background:white;border:1px solid var(--line);border-radius:22px;padding:18px 20px;box-shadow:var(--soft-shadow);}
.chart-pill-title{background:var(--green);color:white;border-radius:999px;padding:9px 18px;text-align:center;font-size:13px;font-weight:900;margin:-6px auto 14px auto;max-width:260px;}
.growth-svg{width:100%;height:190px;display:block;}
.axis-label{font-size:10px;fill:#6f7d75;font-weight:700;}
.grid-line{stroke:#E6EFE9;stroke-width:1;}
.day-gauge-row{display:flex;gap:18px;align-items:flex-start;justify-content:space-between;flex-wrap:nowrap;}
.day-gauge-item{flex:1;min-width:82px;text-align:center;}
.mini-gauge{width:94px;height:48px;margin:0 auto;border-radius:94px 94px 0 0;background:conic-gradient(from 270deg,var(--green) 0deg,var(--green) calc(var(--deg,0)*1deg),#ECEFEE calc(var(--deg,0)*1deg),#ECEFEE 180deg);position:relative;overflow:hidden;}
.mini-gauge:after{content:"";position:absolute;left:15px;right:15px;bottom:-32px;height:64px;background:white;border-radius:50%;}
.mini-gauge-value{font-size:26px;font-weight:850;color:var(--deep);margin-top:-31px;position:relative;z-index:2;}
.mini-gauge-label{font-size:11px;color:#5E6C65;font-weight:800;margin-top:15px;}
@media(max-width:1000px){.analytics-extra-grid{grid-template-columns:1fr;}.day-gauge-row{flex-wrap:wrap}.day-gauge-item{min-width:120px;}}


.live-shell{margin-top:18px;}
.live-grid{display:grid;grid-template-columns:.9fr 1.15fr;gap:24px;align-items:start;}
.live-stack{display:flex;flex-direction:column;gap:18px;}
.live-card{background:#fff;border:1px solid rgba(5,31,32,.08);border-radius:22px;padding:24px;box-shadow:0 10px 28px rgba(5,31,32,.06);}
.live-card-header{display:flex;align-items:center;gap:16px;margin-bottom:22px;}
.live-icon{width:48px;height:48px;border-radius:16px;background:#EAF7EE;color:#235347;display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:900;}
.live-card-title{font-size:20px;font-weight:950;color:#051F20;letter-spacing:-.03em;}
.live-card-sub{font-size:13px;color:#64736B;margin-top:4px;font-weight:600;}
.live-setting-block{padding:14px 0;border-bottom:1px solid rgba(5,31,32,.08);}
.live-setting-block:last-child{border-bottom:none;}
.live-setting-top{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:8px;}
.live-setting-name{font-size:14px;font-weight:900;color:#0B2B26;}
.live-setting-desc{font-size:12px;color:#64736B;margin-top:4px;}
.live-setting-value{font-size:14px;color:#006B3C;font-weight:950;}
.live-note{background:linear-gradient(135deg,#F1FBF4,#FFFFFF);border:1px solid #D7ECE0;border-radius:14px;padding:14px 16px;display:flex;gap:12px;align-items:flex-start;color:#0B2B26;font-size:13px;line-height:1.55;margin-top:12px;}
.live-note-icon{width:26px;height:26px;border-radius:50%;border:1px solid #8EB69B;color:#235347;display:flex;align-items:center;justify-content:center;font-weight:900;flex:0 0 auto;}


.stSlider [data-baseweb="slider"] div{border-color:#235347 !important;}
.stSlider [data-baseweb="slider"] [role="slider"]{background:#235347 !important;border-color:#235347 !important;box-shadow:0 0 0 3px rgba(35,83,71,.12) !important;}
.stSlider [data-baseweb="slider"] > div > div{background:#E3EAE6 !important;}
.stSlider [data-baseweb="slider"] > div > div > div{background:#235347 !important;}
input[type="range"]{accent-color:#235347 !important;}


.upload-zone .live-icon{background:#EAF7EE !important;color:#235347 !important;}
.upload-zone [data-testid="stFileUploader"]{background:#FCFEFD !important;border:1px dashed rgba(35,83,71,.26) !important;border-radius:16px !important;padding:18px !important;box-shadow:inset 0 0 0 1px rgba(255,255,255,.5);}
.upload-zone [data-testid="stFileUploader"] section{border:0 !important;background:transparent !important;padding:12px !important;}
.upload-zone [data-testid="stFileUploader"] button{background:white !important;color:#235347 !important;border:1px solid rgba(35,83,71,.18) !important;border-radius:12px !important;font-weight:800 !important;box-shadow:none !important;}
.upload-zone [data-testid="stFileUploader"] small{color:#64736B !important;}


.camera-card{min-height:auto;padding-bottom:24px;}
.camera-holder{background:#171A1F;border-radius:18px;padding:18px;margin-top:8px;box-shadow:inset 0 0 0 1px rgba(255,255,255,.05);}
.camera-holder [data-testid="stCameraInput"]{background:transparent !important;border:0 !important;border-radius:18px !important;padding:0 !important;box-shadow:none !important;}
.camera-holder [data-testid="stCameraInput"] label{display:none !important;}
.camera-holder [data-testid="stCameraInput"] video,
.camera-holder [data-testid="stCameraInput"] img{border-radius:16px !important;min-height:420px !important;width:100% !important;object-fit:cover !important;background:#171A1F !important;display:block !important;}
.camera-holder [data-testid="stCameraInput"] button{background:#EAF7EE !important;color:#006B3C !important;border:1px solid rgba(35,83,71,.12) !important;border-radius:14px !important;height:58px !important;font-size:16px !important;font-weight:900 !important;margin-top:18px !important;box-shadow:none !important;width:100% !important;}
.camera-holder [data-testid="stCameraInput"] button:hover{background:#DAF1DE !important;color:#0B2B26 !important;}
.live-tip{margin-top:24px;background:linear-gradient(135deg,#F0FAF3,#FFFFFF);border:1px solid rgba(5,31,32,.06);border-radius:16px;padding:18px 22px;color:#526159;font-size:14px;font-weight:600;}
.live-tip b{color:#0B2B26;margin-right:6px;}
.live-results{margin-top:22px;}
.live-result-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:14px;}
.live-result-card{background:white;border:1px solid rgba(5,31,32,.08);border-radius:18px;padding:16px;box-shadow:0 8px 20px rgba(5,31,32,.05);}
@media(max-width:1000px){.live-grid{grid-template-columns:1fr}.camera-holder [data-testid="stCameraInput"] video,.camera-holder [data-testid="stCameraInput"] img{min-height:260px !important}.live-result-grid{grid-template-columns:1fr}}


.live-grid{display:grid;grid-template-columns:.78fr 1.22fr;gap:26px;align-items:stretch;}
.live-stack,.live-left-panel,.camera-card{height:100%;}
.live-left-panel{padding:26px 26px 18px 26px;}
.live-card-header{margin-bottom:18px;}
.live-setting-block{padding:12px 0 8px 0;border-bottom:none;}
.live-left-panel .stSlider{margin-bottom:20px;}
.live-left-panel [data-testid="stSlider"]{padding-top:0 !important;}
.live-note{margin-top:6px;margin-bottom:18px;}
.upload-inline{margin-top:12px;padding-top:20px;border-top:1px solid rgba(5,31,32,.08);}
.upload-inline .live-card-header{background:#fff;border:1px solid rgba(5,31,32,.08);border-radius:22px;padding:18px;margin-bottom:0;box-shadow:0 8px 20px rgba(5,31,32,.04);}
.upload-inline [data-testid="stFileUploader"]{background:#F1F5F6 !important;border:0 !important;border-radius:14px !important;padding:12px 14px !important;margin-top:0 !important;}
.upload-inline [data-testid="stFileUploader"] section{background:transparent !important;border:0 !important;padding:0 !important;}
.upload-inline [data-testid="stFileUploader"] button{background:#fff !important;color:#235347 !important;border:1px solid rgba(35,83,71,.16) !important;border-radius:10px !important;height:38px !important;font-weight:800 !important;}
.camera-card{display:flex;flex-direction:column;min-height:720px;padding:26px;}
.camera-card .camera-holder{flex:1;display:flex;flex-direction:column;min-height:600px;background:#171A1F;border-radius:18px;padding:18px;margin-top:10px;}
.camera-holder [data-testid="stCameraInput"]{flex:1;display:flex;flex-direction:column;}
.camera-holder [data-testid="stCameraInput"] > div{flex:1;}
.camera-holder [data-testid="stCameraInput"] video,
.camera-holder [data-testid="stCameraInput"] img{min-height:560px !important;height:560px !important;width:100% !important;border-radius:16px !important;object-fit:cover !important;background:#171A1F !important;}
.camera-holder [data-testid="stCameraInput"] button{background:#EAF7EE !important;color:#235347 !important;border:1px solid #D5EBDD !important;border-radius:14px !important;height:60px !important;font-size:16px !important;font-weight:950 !important;margin-top:18px !important;width:100% !important;box-shadow:none !important;}
.camera-holder [data-testid="stCameraInput"] button:hover{background:#DAF1DE !important;color:#0B2B26 !important;}
.camera-holder [data-testid="stCameraInput"] button:before{content:"";font-size:18px;margin-right:8px;}
@media(max-width:1000px){.live-grid{grid-template-columns:1fr}.camera-card{min-height:auto}.camera-card .camera-holder{min-height:auto}.camera-holder [data-testid="stCameraInput"] video,.camera-holder [data-testid="stCameraInput"] img{height:320px !important;min-height:320px !important;}}


.live-grid{display:grid;grid-template-columns:.78fr 1.22fr;gap:26px;align-items:start;}
.live-card{background:#fff;border:1px solid rgba(5,31,32,.08);border-radius:22px;padding:24px;box-shadow:0 10px 28px rgba(5,31,32,.06);}
.live-left-panel{padding:26px;}
.camera-card-fixed{background:#fff;border:1px solid rgba(5,31,32,.08);border-radius:22px;padding:26px;box-shadow:0 10px 28px rgba(5,31,32,.06);min-height:720px;}
.camera-card-fixed [data-testid="stCameraInput"]{background:#171A1F !important;border:0 !important;border-radius:18px !important;padding:18px !important;box-shadow:none !important;margin-top:14px !important;}
.camera-card-fixed [data-testid="stCameraInput"] label{display:none !important;}
.camera-card-fixed [data-testid="stCameraInput"] video,
.camera-card-fixed [data-testid="stCameraInput"] img{height:560px !important;min-height:560px !important;width:100% !important;object-fit:cover !important;border-radius:16px !important;background:#171A1F !important;display:block !important;}
.camera-card-fixed [data-testid="stCameraInput"] button{background:#EAF7EE !important;color:#235347 !important;border:1px solid #D5EBDD !important;border-radius:14px !important;height:58px !important;font-size:16px !important;font-weight:950 !important;margin-top:18px !important;width:100% !important;box-shadow:none !important;}
.camera-card-fixed [data-testid="stCameraInput"] button:hover{background:#DAF1DE !important;color:#0B2B26 !important;}
.camera-card-fixed [data-testid="stCameraInput"] button:before{content:"";font-size:17px;margin-right:8px;}
.live-left-panel [data-testid="stSlider"] div[data-baseweb="slider"] div{background-color:#235347;}
.live-left-panel [role="slider"]{background:#235347 !important;border-color:#235347 !important;box-shadow:none !important;}
.upload-box-fixed [data-testid="stFileUploader"]{background:#F3F6F8 !important;border-radius:14px !important;border:1px solid rgba(5,31,32,.06) !important;padding:16px !important;}
.upload-box-fixed [data-testid="stFileUploader"] button{background:white !important;color:#235347 !important;border:1px solid #D5EBDD !important;border-radius:10px !important;}
@media(max-width:1000px){.live-grid{grid-template-columns:1fr}.camera-card-fixed{min-height:auto}.camera-card-fixed [data-testid="stCameraInput"] video,.camera-card-fixed [data-testid="stCameraInput"] img{height:320px !important;min-height:320px !important;}}


.live-final-shell{
    margin-top:18px;
}
.live-final-shell [data-testid="stVerticalBlockBorderWrapper"]{
    border:1px solid rgba(5,31,32,.08) !important;
    border-radius:22px !important;
    box-shadow:0 10px 28px rgba(5,31,32,.06) !important;
    background:#fff !important;
    padding:22px !important;
}
.final-card-header,
.camera-header-final,
.upload-header-final{
    margin-bottom:20px !important;
}
.final-note{
    margin-top:16px !important;
    margin-bottom:0 !important;
}
.live-final-shell [data-testid="stSlider"]{
    margin-bottom:18px !important;
}
.live-final-shell [data-testid="stSlider"] [data-baseweb="slider"] > div > div,
.live-final-shell [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div{
    background:#235347 !important;
}
.live-final-shell [data-testid="stSlider"] [role="slider"]{
    background:#235347 !important;
    border-color:#235347 !important;
    box-shadow:0 0 0 3px rgba(35,83,71,.12) !important;
}
.live-final-shell [data-testid="stFileUploader"]{
    background:#F3F6F8 !important;
    border:1px solid rgba(5,31,32,.06) !important;
    border-radius:14px !important;
    padding:14px !important;
    margin-top:0 !important;
}
.live-final-shell [data-testid="stFileUploader"] section{
    background:transparent !important;
    border:0 !important;
    padding:0 !important;
}
.live-final-shell [data-testid="stFileUploader"] button{
    background:#fff !important;
    color:#235347 !important;
    border:1px solid #D5EBDD !important;
    border-radius:10px !important;
    height:38px !important;
    font-weight:850 !important;
    box-shadow:none !important;
}
.live-final-shell [data-testid="stCameraInput"]{
    background:#171A1F !important;
    border:0 !important;
    border-radius:18px !important;
    padding:18px !important;
    box-shadow:none !important;
    margin-top:10px !important;
}
.live-final-shell [data-testid="stCameraInput"] label{
    display:none !important;
}
.live-final-shell [data-testid="stCameraInput"] video,
.live-final-shell [data-testid="stCameraInput"] img{
    height:520px !important;
    min-height:520px !important;
    width:100% !important;
    object-fit:cover !important;
    border-radius:16px !important;
    background:#171A1F !important;
    display:block !important;
}
.live-final-shell [data-testid="stCameraInput"] button{
    width:100% !important;
    height:58px !important;
    margin-top:18px !important;
    background:#EAF7EE !important;
    color:#235347 !important;
    border:1px solid #D5EBDD !important;
    border-radius:14px !important;
    font-size:16px !important;
    font-weight:950 !important;
    box-shadow:none !important;
}
.live-final-shell [data-testid="stCameraInput"] button:hover{
    background:#DAF1DE !important;
    color:#0B2B26 !important;
}
.live-final-shell [data-testid="stCameraInput"] button:before{
    content:"";
    font-size:17px;
    margin-right:8px;
}
.final-live-tip{
    margin-top:22px !important;
}
@media(max-width:1000px){
    .live-final-shell [data-testid="stCameraInput"] video,
    .live-final-shell [data-testid="stCameraInput"] img{
        height:320px !important;
        min-height:320px !important;
    }
}


.live-final-shell{
    max-width: 1240px;
    margin-left:auto;
    margin-right:auto;
}
.live-final-shell [data-testid="stVerticalBlockBorderWrapper"]{
    padding:24px !important;
}
.upload-center-wrap{
    width:100%;
    max-width:430px;
    margin:0 auto;
}
.upload-center-wrap [data-testid="stVerticalBlockBorderWrapper"]{
    text-align:left !important;
}
.upload-header-final{
    display:flex !important;
    align-items:center !important;
    gap:14px !important;
    margin-bottom:16px !important;
}
.upload-center-wrap [data-testid="stFileUploader"]{
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
    text-align:center !important;
    min-height:64px !important;
}
.upload-center-wrap [data-testid="stFileUploader"] section{
    width:100% !important;
    display:flex !important;
    justify-content:center !important;
    align-items:center !important;
}
.upload-center-wrap [data-testid="stFileUploader"] button{
    height:42px !important;
    min-width:108px !important;
}
.final-note{
    margin-top:20px !important;
    padding:16px 18px !important;
    line-height:1.65 !important;
    border-radius:16px !important;
}
.live-note-icon{
    margin-top:2px !important;
}
.final-live-tip{
    margin-top:26px !important;
    padding:17px 22px !important;
}
.camera-header-final{
    margin-bottom:18px !important;
}
.live-final-shell [data-testid="stCameraInput"]{
    padding:20px !important;
    border-radius:20px !important;
    min-height:690px !important;
}
.live-final-shell [data-testid="stCameraInput"] video,
.live-final-shell [data-testid="stCameraInput"] img{
    height:610px !important;
    min-height:610px !important;
    border-radius:18px !important;
}
.live-final-shell [data-testid="stCameraInput"] button{
    background:#DAF1DE !important;
    color:#0B2B26 !important;
    border:1px solid #BFE5C8 !important;
    border-radius:16px !important;
    height:66px !important;
    font-size:18px !important;
    font-weight:950 !important;
    margin-top:20px !important;
}
.live-final-shell [data-testid="stCameraInput"] button:hover{
    background:#CBEED3 !important;
    color:#051F20 !important;
    transform:translateY(-1px) !important;
}
@media(max-width:1000px){
    .live-final-shell [data-testid="stCameraInput"]{min-height:auto !important;}
    .live-final-shell [data-testid="stCameraInput"] video,
    .live-final-shell [data-testid="stCameraInput"] img{height:360px !important;min-height:360px !important;}
}


.enroll-step-head{
    display:flex;
    align-items:center;
    gap:14px;
    margin-bottom:22px;
}
.enroll-step-num{
    width:34px;
    height:34px;
    border-radius:50%;
    background:#DAF1DE;
    color:#235347;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    font-size:14px;
    font-weight:950;
    flex:0 0 auto;
}
.enroll-step-title{
    font-size:18px;
    font-weight:950;
    color:#051F20;
    letter-spacing:-.03em;
}
.enroll-step-sub{
    font-size:12px;
    color:#64736B;
    font-weight:650;
    margin-top:4px;
}
.enroll-green-note{
    background:#F2FBF5;
    border:1px solid #CFE8D6;
    color:#0B2B26;
    border-radius:12px;
    padding:12px 14px;
    font-size:12px;
    font-weight:750;
    margin:16px 0;
}
.enroll-action-card{
    background:linear-gradient(135deg,#F3FBF6,#FFFFFF);
    border:1px solid #CFE8D6;
    border-radius:16px;
    padding:18px;
    margin:10px 0 14px 0;
}
.enroll-action-title{
    color:#235347;
    font-size:14px;
    font-weight:950;
    margin-bottom:8px;
}
.enroll-action-sub{
    color:#64736B;
    font-size:12px;
    font-weight:650;
}
.enroll-folder-note{
    background:#EEF6FF;
    border:1px solid #D8E8FF;
    color:#315D9A;
    border-radius:12px;
    padding:14px 16px;
    font-size:13px;
    font-weight:750;
    margin-top:18px;
}


button[data-baseweb="tab"]{
    color:#64736B !important;
    font-weight:750 !important;
}
button[data-baseweb="tab"][aria-selected="true"]{
    color:#235347 !important;
}
button[data-baseweb="tab"][aria-selected="true"] p{
    color:#235347 !important;
}
div[data-baseweb="tab-highlight"]{
    background-color:#8EB69B !important;
    height:3px !important;
}


div[data-testid="stTextInput"] input{
    border:1px solid #CFE8D6 !important;
    border-radius:12px !important;
    background:#FCFEFD !important;
    box-shadow:none !important;
}
div[data-testid="stTextInput"] input:focus{
    border-color:#8EB69B !important;
    box-shadow:0 0 0 3px rgba(142,182,155,.18) !important;
}


[data-testid="stFileUploader"]{
    background:#F3F6F8 !important;
    border:1px dashed rgba(35,83,71,.22) !important;
    border-radius:14px !important;
    padding:16px !important;
}
[data-testid="stFileUploader"] section{
    background:transparent !important;
    border:0 !important;
}
[data-testid="stFileUploader"] button{
    background:white !important;
    color:#235347 !important;
    border:1px solid #CFE8D6 !important;
    border-radius:10px !important;
    font-weight:850 !important;
}


.day-share-card, .day-gauge-row, .day-gauge-item, .mini-gauge, .mini-gauge-value, .mini-gauge-label{display:none !important;}
.growth-chart-card-full{width:100% !important;margin-top:16px !important;padding:18px !important;}
.growth-chart-card-full .growth-svg{height:155px !important;}
.final-note{margin-top:8px !important;margin-bottom:12px !important;padding:7px 10px !important;border-radius:10px !important;font-size:10px !important;line-height:1.25 !important;}
.final-note .live-note-icon{width:18px !important;height:18px !important;font-size:10px !important;margin-top:0 !important;}

.dashboard-search-row{display:none !important;}
.growth-chart-card-full{
    width:100% !important;
    max-width:620px !important;
    margin:14px auto 16px auto !important;
    padding:16px 18px 12px 18px !important;
    border-radius:22px !important;
}
.growth-chart-card-full .chart-pill-title{
    max-width:220px !important;
    padding:8px 16px !important;
    margin:0 auto 6px auto !important;
    font-size:12px !important;
}
.growth-chart-card-full .growth-svg{height:150px !important;}
.day-share-card,.day-gauge-row,.day-gauge-item,.mini-gauge,.mini-gauge-value,.mini-gauge-label{display:none !important;}
.bar-help{margin-top:8px !important;margin-bottom:10px !important;}


.nav-link{
    display:flex;
    align-items:center;
    justify-content:flex-start;
    gap:10px;
    height:40px;
    padding:0 16px;
    margin:4px 0;
    border-radius:14px;
    color:#718078 !important;
    text-decoration:none !important;
    font-size:12px;
    font-weight:700;
    transition:all .18s ease;
    position:relative;
}
.nav-link:hover{
    background:#E6F5EA;
    color:#235347 !important;
    transform:translateX(3px);
}
.nav-link.active{
    background:#DAF1DE;
    color:#0B2B26 !important;
    font-weight:900;
}
.nav-link.active:before{
    content:"";
    position:absolute;
    left:0;
    top:8px;
    width:4px;
    height:24px;
    border-radius:999px;
    background:#235347;
}
.sidebar-status-card{
    margin-top:28px;
    padding:16px;
    border-radius:18px;
    background:#FFFFFF;
    border:1px solid rgba(5,31,32,.08);
    box-shadow:0 10px 24px rgba(5,31,32,.07);
}
.sidebar-status-title{
    font-size:12px;
    font-weight:950;
    color:#0B2B26;
    margin-bottom:4px;
}
.sidebar-status-sub{
    font-size:10px;
    font-weight:700;
    color:#235347;
}
.settings-grid{
    display:grid;
    grid-template-columns:repeat(2,minmax(0,1fr));
    gap:16px;
    margin-top:16px;
}
.settings-card{
    background:#fff;
    border:1px solid rgba(5,31,32,.08);
    border-radius:22px;
    padding:20px;
    box-shadow:0 8px 20px rgba(5,31,32,.06);
}
.settings-title{
    font-size:16px;
    font-weight:950;
    color:#051F20;
    margin-bottom:6px;
}
.settings-sub{
    font-size:12px;
    font-weight:650;
    color:#64736B;
    margin-bottom:14px;
}
.settings-mini-row{
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:10px 0;
    border-bottom:1px solid rgba(5,31,32,.06);
    font-size:12px;
    font-weight:750;
    color:#0B2B26;
}
.settings-mini-row:last-child{border-bottom:0;}
.settings-badge{
    padding:5px 10px;
    border-radius:999px;
    background:#E5F5E9;
    color:#235347;
    font-size:11px;
    font-weight:900;
}
@media(max-width:1000px){.settings-grid{grid-template-columns:1fr;}}


section[data-testid="stSidebar"]{
    background:#FFFFFF !important;
    border-right:1px solid #E9EFEA !important;
    width:260px !important;
    box-shadow:8px 0 28px rgba(5,31,32,.04) !important;
}
section[data-testid="stSidebar"] > div{
    padding:26px 18px 20px 18px !important;
}
.premium-brand{display:flex;align-items:center;gap:13px;margin:4px 0 36px 0;padding:0 8px;}
.premium-logo{width:40px;height:40px;border-radius:13px;background:linear-gradient(135deg,#0B2B26,#235347);color:white;display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:950;box-shadow:0 12px 22px rgba(35,83,71,.18);}
.premium-title{font-size:19px;font-weight:950;line-height:1.05;color:#101828;letter-spacing:-.035em;}
.premium-sub{margin-top:4px;font-size:11px;font-weight:600;color:#667085;}
.premium-section{margin:28px 0 10px 8px;font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:#98A2B3;font-weight:900;}
.premium-nav{height:44px;display:flex;align-items:center;gap:14px;padding:0 14px;margin:5px 0;border-radius:13px;color:#344054 !important;text-decoration:none !important;font-size:14px;font-weight:650;position:relative;transition:all .18s ease;}
.premium-nav:hover{background:#F1FAF4;color:#0B2B26 !important;transform:translateX(2px);}
.premium-nav.active{background:#E6F5EA;color:#0B2B26 !important;font-weight:900;}
.premium-nav.active:before{content:"";position:absolute;left:-3px;top:10px;bottom:10px;width:4px;border-radius:999px;background:#235347;}
.premium-nav-icon{width:22px;height:22px;min-width:22px;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#0B2B26;font-size:15px;font-weight:900;}
.premium-nav.active .premium-nav-icon{color:#235347;}
.premium-status{margin-top:42px;padding:15px 14px;border-radius:17px;border:1px solid #E7EEE9;background:#FFFFFF;box-shadow:0 14px 30px rgba(5,31,32,.07);display:flex;align-items:center;gap:12px;}
.premium-status-icon{width:30px;height:30px;border-radius:10px;display:flex;align-items:center;justify-content:center;background:#235347;color:white;font-weight:950;}
.premium-status-title{font-size:13px;font-weight:950;color:#101828;}
.premium-status-sub{font-size:11px;color:#15803D;font-weight:700;margin-top:2px;}
.premium-status-arrow{margin-left:auto;font-size:24px;color:#0B2B26;line-height:1;}
section[data-testid="stSidebar"] div.stButton,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] hr,
section[data-testid="stSidebar"] .brand-box,
section[data-testid="stSidebar"] .sidebar-status-card,
section[data-testid="stSidebar"] .nav-link{display:none !important;}
.mini-date-real{background:white !important;border:1px solid #E4E7EC !important;border-radius:12px !important;color:#101828 !important;height:44px !important;padding:0 18px !important;font-size:13px !important;font-weight:850 !important;box-shadow:0 8px 18px rgba(5,31,32,.06) !important;}
section.main div.stButton > button{border-radius:12px !important;}
section[data-testid="stSidebar"] [data-testid="stTextInput"],section[data-testid="stSidebar"] [data-testid="stNumberInput"]{display:none !important;}


.premium-title{
    font-size:22px !important;
    font-weight:950 !important;
}
.premium-sub{
    font-size:12px !important;
    font-weight:650 !important;
}
.premium-section{
    font-size:11px !important;
    font-weight:950 !important;
    letter-spacing:.13em !important;
    margin-top:30px !important;
    margin-bottom:12px !important;
}
.premium-nav{
    height:48px !important;
    font-size:16px !important;
    font-weight:800 !important;
    gap:15px !important;
    padding:0 16px !important;
    border-radius:14px !important;
}
.premium-nav-icon{
    width:24px !important;
    min-width:24px !important;
    height:24px !important;
    font-size:17px !important;
    line-height:1 !important;
}
.premium-status-link{
    text-decoration:none !important;
    color:inherit !important;
    cursor:pointer !important;
}
.premium-status{
    margin-top:38px !important;
    padding:18px 16px !important;
    border-radius:19px !important;
    gap:13px !important;
    transition:all .18s ease !important;
}
.premium-status:hover{
    background:#F4FBF6 !important;
    transform:translateY(-1px) !important;
    box-shadow:0 14px 28px rgba(5,31,32,.09) !important;
}
.premium-status-icon{
    width:34px !important;
    height:34px !important;
    border-radius:12px !important;
    font-size:17px !important;
}
.premium-status-title{
    font-size:14px !important;
    font-weight:950 !important;
}
.premium-status-sub{
    font-size:12px !important;
    font-weight:800 !important;
}
.premium-status-arrow{
    margin-left:auto !important;
    font-size:24px !important;
    font-weight:900 !important;
    color:#235347 !important;
}


.premium-nav-icon{
    font-size:11px !important;
    font-weight:950 !important;
    letter-spacing:.02em !important;
    background:#F1FAF4 !important;
    color:#235347 !important;
}
.premium-status-icon{
    font-size:10px !important;
    letter-spacing:.02em !important;
}
.live-icon{
    font-size:16px !important;
}
.stat-corner{
    font-size:15px !important;
}


.mobile-bottom-nav{
    display:none;
}
@media(max-width:768px){
    section[data-testid="stSidebar"]{
        display:none !important;
    }
    [data-testid="collapsedControl"]{
        display:none !important;
    }
    .block-container{
        padding:0.9rem 0.85rem 7rem 0.85rem !important;
        max-width:100% !important;
    }
    section.main > div.block-container{
        border-radius:0 !important;
        box-shadow:none !important;
        border:none !important;
    }
    .title-main{
        font-size:26px !important;
    }
    .title-sub{
        font-size:12px !important;
    }
    .top-user{
        display:none !important;
    }
    .header-buttons{
        width:100% !important;
    }
    .dashboard-grid,
    .dashboard-grid-2,
    .analytics-grid,
    .percent-grid,
    .settings-grid,
    .live-grid{
        grid-template-columns:1fr !important;
    }
    .stat-card,
    .dash-card,
    .analytics-card,
    .live-card{
        border-radius:18px !important;
    }
    .mobile-bottom-nav{
        position:fixed;
        left:10px;
        right:10px;
        bottom:10px;
        z-index:999999;
        display:flex;
        align-items:center;
        gap:8px;
        padding:9px;
        background:rgba(255,255,255,.96);
        border:1px solid #E5ECE7;
        border-radius:22px;
        box-shadow:0 14px 36px rgba(5,31,32,.18);
        overflow-x:auto;
        -webkit-overflow-scrolling:touch;
    }
    .mobile-bottom-nav::-webkit-scrollbar{
        display:none;
    }
    .mobile-nav-item{
        min-width:76px;
        height:58px;
        border-radius:16px;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        gap:4px;
        color:#667085 !important;
        text-decoration:none !important;
        font-size:10px;
        font-weight:850;
        white-space:nowrap;
        transition:all .18s ease;
    }
    .mobile-nav-item:active{
        transform:scale(.96);
    }
    .mobile-nav-item.active{
        background:#DAF1DE;
        color:#0B2B26 !important;
    }
    .mobile-nav-code{
        width:24px;
        height:24px;
        border-radius:9px;
        display:flex;
        align-items:center;
        justify-content:center;
        background:#F1FAF4;
        color:#235347;
        font-size:10px;
        font-weight:950;
        letter-spacing:.02em;
    }
    .mobile-nav-item.active .mobile-nav-code{
        background:#235347;
        color:white;
    }
    .mobile-nav-label{
        font-size:10px;
        line-height:1;
    }
}


@media(max-width:768px){
    .mobile-bottom-nav{
        left:12px !important;
        right:12px !important;
        bottom:12px !important;
        height:76px !important;
        max-width:calc(100vw - 24px) !important;
        overflow-x:auto !important;
        overflow-y:hidden !important;
        z-index:2147483647 !important;
        background:#FFFFFF !important;
    }
    .mobile-nav-item{
        flex:0 0 auto !important;
    }
    .block-container{
        padding-bottom:7.5rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    mtcnn_all = MTCNN(
        image_size=IMAGE_SIZE,
        margin=0,
        keep_all=True,
        post_process=True,
        min_face_size=MIN_FACE_SIZE,
        device=DEVICE,
    )

    mtcnn_one = MTCNN(
        image_size=IMAGE_SIZE,
        margin=0,
        keep_all=False,
        post_process=True,
        min_face_size=MIN_FACE_SIZE,
        device=DEVICE,
    )

    model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
    return mtcnn_all, mtcnn_one, model


def get_qdrant_client():
    return QdrantClient(
        host=st.session_state.qdrant_host,
        port=int(st.session_state.qdrant_port),
    )


def qdrant_ready():
    try:
        client = get_qdrant_client()
        client.get_collections()
        return True, ""
    except Exception as e:
        return False, str(e)


def ensure_attendance_collection(client):
    """Create the attendance_events collection if it does not exist."""
    collections = client.get_collections().collections
    existing = [c.name for c in collections]

    if ATTENDANCE_COLLECTION not in existing:
        client.create_collection(
            collection_name=ATTENDANCE_COLLECTION,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )


def save_attendance_event_to_qdrant(client, embedding, decision, crop_path, face_num):
    """
    Save recognition/attendance event to Qdrant.

    Important:
    - tryface = enrolled student database
    - attendance_events = attendance recognition logs/results
    """
    ensure_attendance_collection(client)

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding.tolist(),
        payload={
            "student_name": decision["name"],
            "status": decision["status"],
            "score": float(decision["top1"]),
            "gap": float(decision["gap"]),
            "face_number": int(face_num),
            "crop_path": str(crop_path),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "attendance_website",
        },
    )

    client.upsert(
        collection_name=ATTENDANCE_COLLECTION,
        points=[point],
    )


def count_attendance_events():
    try:
        client = get_qdrant_client()
        result = client.count(collection_name=ATTENDANCE_COLLECTION, exact=True)
        return int(result.count)
    except Exception:
        return 0


def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def expand_box(box, w, h, scale=EXTRA_BOX_SCALE):
    """
    Tighter crop for the website.
    This focuses mainly on the face area and removes most hair/clothes/background.
    """
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    cy = cy - 0.03 * bh

    new_w = bw * (1.0 + scale)
    new_h = bh * (1.0 + scale)

    nx1 = clamp(int(cx - new_w / 2.0), 0, w - 1)
    ny1 = clamp(int(cy - new_h / 2.0), 0, h - 1)
    nx2 = clamp(int(cx + new_w / 2.0), 0, w - 1)
    ny2 = clamp(int(cy + new_h / 2.0), 0, h - 1)

    if nx2 <= nx1:
        nx2 = clamp(nx1 + 1, 0, w - 1)
    if ny2 <= ny1:
        ny2 = clamp(ny1 + 1, 0, h - 1)

    return nx1, ny1, nx2, ny2


def ellipse_mask(h, w):
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(
        m,
        (w // 2, h // 2),
        (int(w * 0.44), int(h * 0.58)),
        0,
        0,
        360,
        255,
        -1,
    )
    return m


def feather_mask(mask, feather=3):
    if feather <= 0:
        return mask
    k = feather * 2 + 1
    return cv2.GaussianBlur(mask, (k, k), 0)


def composite_on_white(img_bgr, mask_soft):
    m = (mask_soft.astype(np.float32) / 255.0)[..., None]
    white = np.full_like(img_bgr, 255, dtype=np.uint8)
    return (img_bgr.astype(np.float32) * m + white.astype(np.float32) * (1 - m)).astype(np.uint8)


def force_white_background(face_bgr_160):
    m = feather_mask(ellipse_mask(IMAGE_SIZE, IMAGE_SIZE), FEATHER)
    return composite_on_white(face_bgr_160, m)


def face_tensor_to_bgr160(face_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert MTCNN aligned tensor to normal BGR image.
    Handles both tensor ranges: [-1, 1] and [0, 1]. This fixes dark Face details crops.
    """
    face_rgb = face_tensor.permute(1, 2, 0).detach().cpu().numpy()

    if face_rgb.min() < 0:
        face_rgb = (face_rgb + 1.0) / 2.0

    face_rgb = (face_rgb * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)


def face_bgr_to_embedding(face_bgr_160: np.ndarray, model) -> np.ndarray:
    rgb = cv2.cvtColor(face_bgr_160, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1)
    x = (x - 0.5) / 0.5
    x = x.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(x).detach().cpu().numpy()[0].astype(np.float32)

    return l2_normalize(emb)


def qdrant_search(client, emb: np.ndarray):
    res = client.query_points(
        collection_name=st.session_state.collection,
        query=emb.tolist(),
        limit=int(st.session_state.top_k),
        with_payload=True,
    )
    return res.points


def resolve_image_path(image_path):
    """
    Return a usable local image path from Qdrant payload.

    Qdrant payload should store one of these keys:
    - image_path
    - source_image
    - crop_path

    This helper also tries common project folders if the stored path is relative
    or if the app is opened from a different working directory.
    """
    if not image_path:
        return None

    path = Path(str(image_path))

    if path.exists():
        return str(path)

    candidate = Path.cwd() / path
    if candidate.exists():
        return str(candidate)

    common_folders = [
        CAPTURE_DIR,
        CLEAN_FACE_DIR,
        MY_DATASET_DIR,
        Path("."),
    ]

    for folder in common_folders:
        candidate = folder / path.name
        if candidate.exists():
            return str(candidate)

    for folder in [CAPTURE_DIR, CLEAN_FACE_DIR, MY_DATASET_DIR]:
        if folder.exists():
            matches = list(folder.rglob(path.name))
            if matches:
                return str(matches[0])

    return None


def decide_name(points):
    if not points:
        return {
            "name": "UNKNOWN",
            "status": "UNKNOWN",
            "top1": -1.0,
            "top2": -1.0,
            "gap": 0.0,
            "candidates": [],
            "distinct": [],
            "best_image_path": None,
        }

    scored = []
    best_by_name = {}

    for p in points:
        payload = p.payload or {}
        name = payload.get("student_name", "UNKNOWN_NAME")
        score = float(p.score)
        image_path = payload.get("image_path") or payload.get("source_image") or payload.get("crop_path")

        scored.append({
            "candidate_name": name,
            "cosine_score": score,
            "image_path": image_path,
        })

        if name not in best_by_name or score > best_by_name[name]["score"]:
            best_by_name[name] = {
                "score": score,
                "image_path": image_path,
            }

    scored.sort(key=lambda x: x["cosine_score"], reverse=True)

    distinct = []
    for name, data in best_by_name.items():
        distinct.append({
            "person": name,
            "best_cosine": data["score"],
            "image_path": data["image_path"],
        })

    distinct.sort(key=lambda x: x["best_cosine"], reverse=True)

    top1_name = distinct[0]["person"]
    top1 = distinct[0]["best_cosine"]
    best_image_path = distinct[0].get("image_path")
    top2 = distinct[1]["best_cosine"] if len(distinct) > 1 else -1.0
    gap = top1 - top2 if top2 >= 0 else 1.0

    result = {
        "name": top1_name,
        "status": "UNKNOWN",
        "top1": top1,
        "top2": top2,
        "gap": gap,
        "candidates": scored,
        "distinct": distinct,
        "best_image_path": best_image_path,
    }

    if top1 >= st.session_state.threshold and gap >= st.session_state.gap:
        result["status"] = "KNOWN"
    elif top1 >= st.session_state.threshold:
        result["status"] = "MANUAL_CHECK"
    else:
        result["name"] = "UNKNOWN"
        result["status"] = "UNKNOWN"

    return result


def recognize_image(image_pil):
    mtcnn_all, mtcnn_one, model = load_models()
    client = get_qdrant_client()

    image_pil = image_pil.convert("RGB")
    frame_rgb = np.array(image_pil)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    h, w = frame_bgr.shape[:2]
    boxes, probs, _ = mtcnn_all.detect(image_pil, landmarks=True)

    results = []
    annotated = frame_bgr.copy()

    if boxes is None or probs is None or len(boxes) == 0:
        return results, annotated

    valid_idxs = [i for i in range(len(boxes)) if float(probs[i]) >= PROB_THRESHOLD]

    for face_num, idx in enumerate(valid_idxs, 1):
        x1, y1, x2, y2 = expand_box(boxes[idx], w, h)
        crop_bgr = frame_bgr[y1:y2, x1:x2]

        if crop_bgr.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        aligned_tensor = mtcnn_one(crop_rgb)

        if aligned_tensor is None:
            face_bgr_160 = cv2.resize(crop_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        else:
            face_bgr_160 = face_tensor_to_bgr160(aligned_tensor)

        if FORCE_WHITE_BG:
            face_bgr_160 = force_white_background(face_bgr_160)

        emb = face_bgr_to_embedding(face_bgr_160, model)
        points = qdrant_search(client, emb)
        decision = decide_name(points)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = decision["name"].replace(" ", "_")
        crop_path = CAPTURE_DIR / f"{ts}_{safe_name}_face{face_num}.jpg"
        cv2.imwrite(str(crop_path), face_bgr_160)

        decision["face_num"] = face_num
        decision["crop_path"] = str(crop_path)

        save_attendance_event_to_qdrant(
            client=client,
            embedding=emb,
            decision=decision,
            crop_path=crop_path,
            face_num=face_num,
        )

        results.append(decision)

        if decision["status"] == "KNOWN":
            color = (0, 130, 60)
            label = f"PRESENT: {decision['name']}"
        elif decision["status"] == "MANUAL_CHECK":
            color = (0, 150, 200)
            label = "MANUAL REVIEW"
        else:
            color = (0, 0, 200)
            label = "UNKNOWN"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    return results, annotated


def load_attendance():
    columns = ["name", "status", "entry_time", "leave_time", "last_seen", "best_score"]

    if os.path.exists(ATTENDANCE_CSV):
        df = pd.read_csv(ATTENDANCE_CSV, dtype=str)
    else:
        df = pd.DataFrame(columns=columns)

    for col in columns:
        if col not in df.columns:
            df[col] = ""

    df = df.fillna("")
    return df[columns]


def weekly_attendance_counts(df: pd.DataFrame):
    """Return realistic weekly attendance counts from the CSV/Excel-style attendance log."""
    days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    counts = {day: 0 for day in days}

    if df is None or df.empty:
        return [(day, 0, 8, False) for day in days]

    date_source = None
    for col in ["entry_time", "last_seen", "leave_time"]:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                date_source = parsed
                break

    if date_source is None:
        return [(day, 0, 8, False) for day in days]

    for dt in date_source.dropna():
        idx = int(dt.weekday())
        day = days[(idx + 1) % 7]
        counts[day] += 1

    max_count = max(counts.values()) if counts else 0
    result = []
    for day in days:
        count = counts[day]
        height = 8 if max_count == 0 else max(18, int((count / max_count) * 100))
        is_light = count < max_count and count != 0
        result.append((day, count, height, is_light))
    return result


def weekly_bars_html(df: pd.DataFrame):
    bars = []
    for day, count, height, is_light in weekly_attendance_counts(df):
        cls = "bar-fill light" if is_light else "bar-fill"
        bars.append(
            f"""<div class="bar-wrap">
                    <div class="bar"><div class="{cls}" style="height:{height}%"></div></div>
                    <div class="bar-label">{day}</div>
                    <div class="bar-count">{count}</div>
                </div>"""
        )
    return "".join(bars)


def _attendance_date_series(df: pd.DataFrame):
    """Choose the best real date column from the attendance CSV."""
    if df is None or df.empty:
        return pd.Series(dtype="datetime64[ns]")
    for col in ["entry_time", "last_seen", "leave_time"]:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                return parsed
    return pd.Series(dtype="datetime64[ns]")


def attendance_growth_svg(df: pd.DataFrame):
    """Create a compact, non-stretched SVG line chart from cumulative attendance records."""
    dates = _attendance_date_series(df).dropna()

    width, height = 620, 185
    plot_x, plot_y = 58, 22
    plot_w, plot_h = 520, 105

    if dates.empty:
        labels = []
        max_v = 1
        points = [(plot_x, plot_y + plot_h), (plot_x + plot_w, plot_y + plot_h)]
    else:
        daily = dates.dt.date.value_counts().sort_index()
        cumulative = daily.cumsum()
        values = cumulative.tolist()
        labels = [pd.to_datetime(d).strftime("%b %d") for d in cumulative.index]
        max_v = max(values) if values else 1

        points = []
        for i, val in enumerate(values):
            x = plot_x + (plot_w * i / max(len(values) - 1, 1))
            y = plot_y + plot_h - ((val / max_v) * plot_h)
            points.append((x, y))

        if len(points) == 1:
            points.append((plot_x + plot_w, points[0][1]))

    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#235347" />'
        for x, y in points
    )

    y_ticks = [0, max_v // 2 if max_v > 1 else 0, max_v]
    y_ticks = sorted(set(y_ticks))
    grid = ""
    for tick in y_ticks:
        y = plot_y + plot_h - ((tick / max_v) * plot_h if max_v else 0)
        grid += f'<line class="grid-line" x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" />'
        grid += f'<text class="axis-label" x="18" y="{y + 4:.1f}">{tick}</text>'

    xlabels = ""
    if labels:
        label_indexes = list(range(len(labels)))
        if len(labels) > 4:
            label_indexes = sorted(set([0, len(labels)//3, (len(labels)*2)//3, len(labels)-1]))
        for i in label_indexes:
            x, _ = points[i]
            xlabels += f'<text class="axis-label" text-anchor="middle" x="{x:.1f}" y="160">{labels[i]}</text>'

    return f"""
    <svg class="growth-svg" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet">
        {grid}
        <line class="grid-line" x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" />
        <line class="grid-line" x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" />
        <polyline points="{poly}" fill="none" stroke="#235347" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
        {circles}
        {xlabels}
    </svg>
    """


def day_share_gauges_html(df: pd.DataFrame):
    """Create weekday share gauges from the real attendance dates."""
    weekly = weekly_attendance_counts(df)
    total = sum(count for _, count, _, _ in weekly)
    items = []
    for day, count, _, _ in weekly:
        pct = int(round((count / total) * 100)) if total else 0
        deg = int((pct / 100) * 180)
        items.append(f"""
            <div class="day-gauge-item">
                <div class="mini-gauge" style="--deg:{deg};"></div>
                <div class="mini-gauge-value">{pct}%</div>
                <div class="mini-gauge-label">{day}</div>
            </div>
        """)
    return "".join(items)


def filter_by_weekday(df: pd.DataFrame, day: str | None) -> pd.DataFrame:
    """Filter attendance rows by the selected weekly attendance bar day."""
    if df is None or df.empty or not day:
        return df

    day_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    target = day_map.get(day)
    if target is None:
        return df

    date_source = None
    for col in ["entry_time", "last_seen", "leave_time"]:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                date_source = parsed
                break

    if date_source is None:
        return df.iloc[0:0]

    return df[date_source.dt.weekday == target]


def filter_dashboard_df(df: pd.DataFrame, query: str | None = None) -> pd.DataFrame:
    """Search across every column in the attendance table: name, status, times, score, dates, etc.
    Multiple words work too: every typed word must appear somewhere in the row."""
    if df is None or df.empty:
        return df

    if query is None:
        query = st.session_state.get("dashboard_search", "")

    query = str(query).strip().lower()
    if not query:
        return df

    search_df = df.fillna("").astype(str)
    row_text = search_df.apply(
        lambda row: " ".join([str(col) for col in search_df.columns] + row.tolist()).lower(),
        axis=1,
    )

    tokens = [t for t in query.replace(",", " ").split() if t]
    mask = pd.Series(True, index=df.index)
    for token in tokens:
        mask = mask & row_text.str.contains(token, case=False, regex=False, na=False)

    return df[mask]

def gauge_degrees(percent: int):
    percent = max(0, min(100, int(percent)))
    return int((percent / 100) * 180)

def save_attendance(df):
    df.to_csv(ATTENDANCE_CSV, index=False)


def mark_present(name, score):
    if name == "UNKNOWN":
        return

    df = load_attendance()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if "leave_time" not in df.columns:
        df["leave_time"] = ""

    if name in df["name"].values:
        df.loc[df["name"] == name, "status"] = "Present"
        df.loc[df["name"] == name, "last_seen"] = now
        df.loc[df["name"] == name, "best_score"] = str(round(float(score), 3))
        if "entry_time" in df.columns:
            empty_entry = df.loc[df["name"] == name, "entry_time"].isna() | (df.loc[df["name"] == name, "entry_time"] == "")
            if empty_entry.any():
                df.loc[df["name"] == name, "entry_time"] = now
    else:
        new_row = {
            "name": name,
            "status": "Present",
            "entry_time": now,
            "leave_time": "",
            "last_seen": now,
            "best_score": str(round(float(score), 3)),
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    save_attendance(df)


def mark_left(name):
    df = load_attendance()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if name in df["name"].values:
        df.loc[df["name"] == name, "status"] = "Left"
        df.loc[df["name"] == name, "leave_time"] = now
        df.loc[df["name"] == name, "last_seen"] = now
        save_attendance(df)


def list_people_in_qdrant():
    client = get_qdrant_client()
    names = []
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=st.session_state.collection,
            limit=100,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )

        for p in points:
            payload = p.payload or {}
            name = payload.get("student_name")
            if name:
                names.append(name)

        if next_offset is None:
            break

    return sorted(set(names)), len(names)


def run_command(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def run_first_available_script(possible_scripts):
    """Run the first script that exists in the app folder."""
    for script_name in possible_scripts:
        script_path = Path(script_name)
        if script_path.exists():
            return run_command(f'python "{script_name}"'), script_name

    missing = ", ".join(possible_scripts)
    return (1, "", f"Could not find any of these scripts in this folder: {missing}"), None


def get_student_image_files(folder: Path):
    """Return supported image files from one student folder only."""
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return sorted(set(files))


def ensure_face_collection(client):
    """Create the face collection if it does not exist. This does NOT delete old data."""
    collections = client.get_collections().collections
    existing = [c.name for c in collections]

    if st.session_state.collection not in existing:
        client.create_collection(
            collection_name=st.session_state.collection,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )


def preprocess_one_student_to_clean_faces(student_name: str):
    """
    Preprocess only one student from my_dataset/student_name
    and save clean faces into clean_faces_4_white/student_name.
    """
    student_name = student_name.strip()
    src_dir = MY_DATASET_DIR / student_name
    out_dir = CLEAN_FACE_DIR / student_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        return False, f"Student folder not found: {src_dir}", 0

    image_paths = get_student_image_files(src_dir)
    if not image_paths:
        return False, f"No images found inside: {src_dir}", 0

    _, mtcnn_one, _ = load_models()
    saved = 0

    for img_path in image_paths:
        try:
            img_pil = Image.open(img_path).convert("RGB")
            aligned_tensor = mtcnn_one(img_pil)

            if aligned_tensor is None:
                img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                face_bgr_160 = cv2.resize(img_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            else:
                face_bgr_160 = face_tensor_to_bgr160(aligned_tensor)

            if FORCE_WHITE_BG:
                face_bgr_160 = force_white_background(face_bgr_160)

            out_path = out_dir / f"{img_path.stem}_clean.jpg"
            cv2.imwrite(str(out_path), face_bgr_160)
            saved += 1

        except Exception as e:
            st.warning(f"Skipped {img_path.name}: {e}")

    if saved == 0:
        return False, "No clean faces were saved. Try a clearer face image.", 0

    return True, f"Saved {saved} clean face(s) for {student_name} only.", saved


def delete_existing_student_points(client, student_name: str):
    """Optional: remove old vectors for one student only before saving new ones."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client.delete(
        collection_name=st.session_state.collection,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="student_name",
                    match=MatchValue(value=student_name),
                )
            ]
        ),
    )


def save_one_student_to_qdrant(student_name: str, replace_old: bool = False):
    """
    Save only one student's clean faces to Qdrant.
    It does NOT recreate the collection and does NOT re-save everyone.
    """
    student_name = student_name.strip()
    student_dir = CLEAN_FACE_DIR / student_name

    if not student_dir.exists():
        return False, f"No clean faces found for {student_name}. Preprocess this student first.", 0

    image_paths = get_student_image_files(student_dir)
    if not image_paths:
        return False, f"No clean face images found inside: {student_dir}", 0

    client = get_qdrant_client()
    ensure_face_collection(client)
    _, _, model = load_models()

    if replace_old:
        delete_existing_student_points(client, student_name)

    points = []
    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        img_bgr = cv2.resize(img_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        emb = face_bgr_to_embedding(img_bgr, model)

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={
                    "student_name": student_name,
                    "image_path": str(img_path),
                    "source_image": str(img_path),
                    "source": "single_student_enrollment",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        )

    if not points:
        return False, "No valid embeddings were created.", 0

    client.upsert(
        collection_name=st.session_state.collection,
        points=points,
    )

    return True, f"Saved only {student_name} to Qdrant ({len(points)} vector(s)).", len(points)


def clean_display_table(data, columns=None, drop_cols=None, round_cols=None):
    """Create a clean dataframe for Streamlit display without long local paths."""
    df = pd.DataFrame(data, columns=columns) if columns else pd.DataFrame(data)

    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if round_cols:
        for col in round_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(3)

    return df


def sidebar():
    """Premium left navigation matching the reference style, with icons and active highlight."""
    icons = {
        "Dashboard": "DB",
        "Analytics": "AN",
        "Live Recognition": "LR",
        "Manual Verification": "MV",
        "Records": "RC",
        "Students": "ST",
        "Enroll Student": "EN",
        "Connections": "CN",
        "Settings": "SE",
    }

    with st.sidebar:
        st.markdown(
            f"""
            <div class="premium-brand">
                <div class="premium-logo">P</div>
                <div>
                    <div class="premium-title">PrivaMin</div>
                    <div class="premium-sub">Attendance System</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pages = {
            "Overview": ["Dashboard", "Analytics"],
            "Attendance": ["Live Recognition", "Manual Verification", "Records"],
            "Roster": ["Students", "Enroll Student"],
            "System": ["Connections", "Settings"],
        }

        current = st.session_state.get("page", "Dashboard")
        for section, items in pages.items():
            st.markdown(f'<div class="premium-section">{section}</div>', unsafe_allow_html=True)
            for item in items:
                active = "active" if current == item else ""
                href = "?nav=" + item.replace(" ", "%20")
                icon = icons.get(item, "")
                nav_html = (
                    f'<a class="premium-nav {active}" href="{href}" target="_self">'
                    f'<span class="premium-nav-icon">{icon}</span>'
                    f'<span>{item}</span>'
                    f'</a>'
                )
                st.markdown(nav_html, unsafe_allow_html=True)

        ok, _ = qdrant_ready()
        status_title = "System Secure" if ok else "System Alert"
        status_sub = "All systems operational" if ok else "Check Qdrant connection"
        st.markdown(
            f"""
            <a class="premium-status premium-status-link" href="?nav=Connections" target="_self" title="Open Connections">
                <div class="premium-status-icon">OK</div>
                <div>
                    <div class="premium-status-title">{status_title}</div>
                    <div class="premium-status-sub">{status_sub}</div>
                </div>
                <div class="premium-status-arrow">&gt;</div>
            </a>
            """,
            unsafe_allow_html=True,
        )

    mobile_pages = [
        ("Dashboard", "DB", "Dashboard"),
        ("Analytics", "AN", "Analytics"),
        ("Live Recognition", "LR", "Live"),
        ("Manual Verification", "MV", "Verify"),
        ("Records", "RC", "Records"),
        ("Students", "ST", "Students"),
        ("Enroll Student", "EN", "Enroll"),
        ("Connections", "CN", "Connect"),
        ("Settings", "SE", "Settings"),
    ]
    mobile_items = []
    current = st.session_state.get("page", "Dashboard")
    for item, code, short_label in mobile_pages:
        active = "active" if current == item else ""
        href = "?nav=" + item.replace(" ", "%20")
        mobile_items.append(
            f'<a class="mobile-nav-item {active}" href="{href}" target="_self">'
            f'<span class="mobile-nav-code">{code}</span>'
            f'<span class="mobile-nav-label">{short_label}</span>'
            f'</a>'
        )
    st.markdown(
        '<div class="mobile-bottom-nav">' + ''.join(mobile_items) + '</div>',
        unsafe_allow_html=True,
    )

def page_header(eyebrow, title, desc):
    today = datetime.now().strftime("%b %d, %Y")
    st.session_state.dashboard_search = ""

    top_l, top_r = st.columns([6, 2])
    with top_l:
        st.markdown("<div style='height:42px;'></div>", unsafe_allow_html=True)
    with top_r:
        st.markdown(
            f"""
            <div class="top-user">
                <div class="user-avatar">P</div>
                <div>
                    <div class="user-name">PrivAIMin</div>
                    <div class="user-mail">Attendance System</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="title-row no-html-buttons">
            <div>
                <div class="title-main">{title}</div>
                <div class="title-sub">{desc}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    h1, h2, h3 = st.columns([7, 1.15, 1.15])
    with h2:
        if st.button("Add Student", use_container_width=True, key=f"header_add_student_{title}"):
            st.session_state.page = "Enroll Student"
            st.rerun()
    with h3:
        st.markdown(f"<div class='mini-date-real'>{today}</div>", unsafe_allow_html=True)

def set_dashboard_view(view):
    """Update only the dashboard detail / recent attendance section."""
    st.session_state.dashboard_view = view


def stat_card(label, value, hint, color="green", view="recent"):
    """Dashboard metric card. The corner arrow stays INSIDE the card and only updates the dashboard detail area."""
    active = "active" if color == "green" else ""
    safe_view = str(view).replace('"', '').replace("'", "")

    st.markdown(
        f"""
        <div class="stat-card {active}">
            <div class="stat-header">
                <div class="stat-label">{label}</div>
                <a class="stat-corner" href="?dash_view={safe_view}#attendance-log" title="Show {label}">&gt;</a>
            </div>
            <div class="stat-value">{value}</div>
            <div class="stat-hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def nav_selector():
     st.markdown("<div style='height:35px;'></div>", unsafe_allow_html=True)
     cols = st.columns([1, 1, 1, 1, 1, 1, 1])
     pages = ["Dashboard", "Live Recognition", "Manual Verification", "Records", "Students", "Enroll Student", "Connections"]
     for col, page in zip(cols, pages):
         with col:
            if st.button(page, use_container_width=True):
              st.session_state.page = page


def page_dashboard():
    page_header(
        "Dashboard",
        "Dashboard",
        "Plan, prioritize, and manage attendance with a clean recognition dashboard.",
    )

    df = load_attendance()
    search_query = ""
    display_df = df
    present_df = df[df["status"].astype(str).str.lower() == "present"] if not df.empty else pd.DataFrame(columns=df.columns)
    left_df = df[df["status"].astype(str).str.lower() == "left"] if not df.empty else pd.DataFrame(columns=df.columns)
    present = len(present_df)
    left = len(left_df)

    try:
        people, total_points = list_people_in_qdrant()
        enrolled = len(people)
    except Exception:
        people, total_points, enrolled = [], 0, 0

    manual_count = len(st.session_state.manual_queue)
    total_for_pct = max(present + left + manual_count, 1)
    attendance_pct = int((present / total_for_pct) * 100)
    left_pct = int((left / total_for_pct) * 100)
    pending_pct = int((manual_count / total_for_pct) * 100)
    weekly_html = weekly_bars_html(df)
    attendance_gauge_deg = gauge_degrees(attendance_pct)

    qdrant_ok, _ = qdrant_ready()
    qdrant_status_pct = 100 if qdrant_ok else 12
    qdrant_status_text = "Online" if qdrant_ok else "Offline"
    csv_status_ok = os.path.exists(ATTENDANCE_CSV)
    csv_status_pct = 100 if csv_status_ok else 12
    csv_status_text = "Ready" if csv_status_ok else "Empty"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        stat_card("Total Students", enrolled, "Total registered students", "green", "students")
    with col2:
        stat_card("Present Today", present, "Updated after each capture", "white", "present")
    with col3:
        stat_card("Left Today", left, "From attendance CSV", "white", "left")
    with col4:
        stat_card("Pending Verify", manual_count, "Manual review queue", "white", "pending")

    st.markdown(
        f"""
        <div class="dashboard-grid clean-dashboard">
            <div class="dash-card">
                <div class="dash-title">Weekly Attendance Bars</div>
                <div class="bar-week">{weekly_html}</div>
            </div>
            <div class="dash-card gauge-card">
                <div class="dash-title">Attendance Percentage</div>
                <div class="gauge" style="--gauge-deg:{attendance_gauge_deg};"></div>
                <div class="gauge-value">{attendance_pct}%</div>
                <div style="font-size:12px;color:#7B8A82;font-weight:800;">Students Present</div>
                <div class="legend"><span>{attendance_pct}% Present</span><span>{left_pct}% Left</span><span>{pending_pct}% Pending</span></div>
            </div>
            <div class="dash-card">
                <div class="dash-title">System Status</div>
                <div class="analytics-row"><div class="analytics-status">Qdrant DB</div><div class="analytics-bar-bg"><div class="analytics-bar-fill" style="width:{qdrant_status_pct}%"></div></div><div class="analytics-count">{qdrant_status_text}</div></div>
                <div class="analytics-row"><div class="analytics-status">FaceNet</div><div class="analytics-bar-bg"><div class="analytics-bar-fill" style="width:100%;background:#8EB69B"></div></div><div class="analytics-count">Ready</div></div>
                <div class="analytics-row"><div class="analytics-status">CSV Log</div><div class="analytics-bar-bg"><div class="analytics-bar-fill" style="width:{csv_status_pct}%;background:#DAF1DE"></div></div><div class="analytics-count">{csv_status_text}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown('<div id="attendance-log" class="analytics-wrap" style="margin-top:16px;">', unsafe_allow_html=True)

    view = st.session_state.dashboard_view
    if view == "students":
        st.markdown('<div class="dashboard-detail-title">Enrolled Students</div><div class="dashboard-detail-sub">Students loaded from Qdrant face collection.</div>', unsafe_allow_html=True)
        if people:
            st.dataframe(pd.DataFrame({"student_name": people}), use_container_width=True, hide_index=True)
        else:
            st.info("No students found in Qdrant, or Qdrant is not connected.")

    elif view == "present":
        st.markdown('<div class="dashboard-detail-title">Present Students</div><div class="dashboard-detail-sub">Students currently marked Present from the attendance CSV/Excel-style log.</div>', unsafe_allow_html=True)
        if present_df.empty:
            st.info("No students are marked Present right now.")
        else:
            st.dataframe(filter_dashboard_df(present_df, search_query), use_container_width=True, hide_index=True)

    elif view == "left":
        st.markdown('<div class="dashboard-detail-title">Left Students</div><div class="dashboard-detail-sub">Students marked Left from the attendance CSV/Excel-style log.</div>', unsafe_allow_html=True)
        if left_df.empty:
            st.info("No students are marked Left right now.")
        else:
            st.dataframe(filter_dashboard_df(left_df, search_query), use_container_width=True, hide_index=True)

    elif view == "pending":
        st.markdown('<div class="dashboard-detail-title">Pending Manual Verification</div><div class="dashboard-detail-sub">Faces waiting for confirmation from the manual queue.</div>', unsafe_allow_html=True)
        if not st.session_state.manual_queue:
            st.info("No pending manual verification items.")
        else:
            pending_rows = []
            for i, item in enumerate(st.session_state.manual_queue, 1):
                pending_rows.append({
                    "#": i,
                    "top_guess": item.get("name", "UNKNOWN"),
                    "score": round(float(item.get("top1", 0)), 3),
                    "gap": round(float(item.get("gap", 0)), 3),
                    "crop_path": item.get("crop_path", ""),
                })
            st.dataframe(pd.DataFrame(pending_rows), use_container_width=True, hide_index=True)

    elif view == "day":
        selected_day = st.session_state.get("dashboard_day")
        day_df = filter_by_weekday(df, selected_day)
        day_df = filter_dashboard_df(day_df, search_query)
        st.markdown(
            f'<div class="dashboard-detail-title">{selected_day or "Selected Day"} Attendance</div>'
            '<div class="dashboard-detail-sub">Attendance records filtered by the weekly bar you clicked.</div>',
            unsafe_allow_html=True,
        )
        if day_df.empty:
            st.info(f"No attendance records found for {selected_day}.")
        else:
            st.dataframe(day_df, use_container_width=True, hide_index=True)

    else:
        st.markdown('<div class="dashboard-detail-title">Attendance Log</div><div class="dashboard-detail-sub">Attendance table returned from the CSV/Excel-style log.</div>', unsafe_allow_html=True)
        if display_df.empty:
            st.info("No matching attendance records found." if st.session_state.dashboard_search else "No attendance records yet. Open Live Recognition and capture a face.")
        else:
            if st.session_state.dashboard_search:
                st.caption(f"Showing {len(display_df)} matching record(s) for: {search_query}")
            st.dataframe(display_df.tail(8), use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Start Live Recognition", use_container_width=True, key="dash_start_live"):
            st.session_state.page = "Live Recognition"
            st.rerun()
    with c2:
        if st.button("Open Manual Verification", use_container_width=True, key="dash_manual"):
            st.session_state.page = "Manual Verification"
            st.rerun()
    with c3:
        if st.button("View Attendance Log", use_container_width=True, key="dash_view_all_records"):
            st.session_state.page = "Records"
            st.session_state.dashboard_view = "recent"
            st.rerun()

def page_live():
    page_header(
        "Live console",
        "Live Recognition",
        "Capture a photo with the browser camera, match it with Qdrant, and mark attendance automatically.",
    )

    ok, msg = qdrant_ready()
    if not ok:
        st.error(f"Qdrant connection failed: {msg}")
        st.stop()

    uploaded = None
    camera_photo = None

    st.markdown('<div class="live-final-shell">', unsafe_allow_html=True)
    settings_col, camera_col = st.columns([0.72, 1.45], gap="large")

    with settings_col:
        with st.container(border=True):
            st.markdown(
                """
                <div class="live-card-header final-card-header">
                    <div class="live-icon">i</div>
                    <div>
                        <div class="live-card-title">Recognition Settings</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="live-setting-block">
                    <div class="live-setting-top">
                        <div>
                            <div class="live-setting-name">Recognition Threshold</div>
                            <div class="live-setting-desc">Adjust the minimum score to consider a match.</div>
                        </div>
                        <div class="live-setting-value">{float(st.session_state.threshold):.2f}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.session_state.threshold = st.slider(
                "Recognition Threshold",
                0.40,
                0.95,
                float(st.session_state.threshold),
                0.01,
                label_visibility="collapsed",
                key="live_threshold_final",
            )

            st.markdown(
                f"""
                <div class="live-setting-block">
                    <div class="live-setting-top">
                        <div>
                            <div class="live-setting-name">Similarity Gap</div>
                            <div class="live-setting-desc">Define the minimum difference between top matches.</div>
                        </div>
                        <div class="live-setting-value">{float(st.session_state.gap):.2f}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.session_state.gap = st.slider(
                "Similarity Gap",
                0.00,
                0.25,
                float(st.session_state.gap),
                0.01,
                label_visibility="collapsed",
                key="live_gap_final",
            )

            st.markdown(
                f"""
                <div class="live-setting-block">
                    <div class="live-setting-top">
                        <div>
                            <div class="live-setting-name">Top K Candidates</div>
                            <div class="live-setting-desc">Number of top candidates to retrieve.</div>
                        </div>
                        <div class="live-setting-value">{int(st.session_state.top_k)}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.session_state.top_k = st.slider(
                "Top K Candidates",
                3,
                20,
                int(st.session_state.top_k),
                1,
                label_visibility="collapsed",
                key="live_topk_final",
            )

            st.markdown(
                """
                <div class="live-note final-note">
                    <div class="live-note-icon">i</div>
                    <div>
                        Current threshold 0.60 for easier matching.<br>
                        Increase it only if wrong matches happen.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

        st.markdown('<div class="upload-center-wrap">', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(
                """
                <div class="live-card-header upload-header-final">
                    <div class="live-icon">UP</div>
                    <div>
                        <div class="live-card-title">Upload Alternative</div>
                        <div class="live-card-sub">Upload an image if camera is not available.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Click to upload or drag and drop",
                type=["jpg", "jpeg", "png"],
                key="live_upload_final",
                label_visibility="collapsed",
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with camera_col:
        with st.container(border=True):
            st.markdown(
                """
                <div class="live-card-header camera-header-final">
                    <div class="live-icon">CAM</div>
                    <div>
                        <div class="live-card-title">Camera</div>
                        <div class="live-card-sub">Use the browser camera. Ensure good lighting for best results.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            camera_photo = st.camera_input(
                "Take Photo",
                key="live_camera_final",
                label_visibility="collapsed",
            )

    st.markdown(
        """
        <div class="live-tip final-live-tip">
            <b>Tip:</b> Ensure the face is clear, well-lit, and centered for better recognition accuracy.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    image_to_process = None
    if camera_photo is not None:
        image_to_process = Image.open(camera_photo)
    elif uploaded is not None:
        image_to_process = Image.open(uploaded)

    if image_to_process is not None:
        st.markdown('<div class="live-results">', unsafe_allow_html=True)
        with st.spinner("Recognizing face..."):
            results, annotated = recognize_image(image_to_process)

        st.markdown('<div class="live-result-grid">', unsafe_allow_html=True)
        left, right = st.columns([1, 1])
        with left:
            st.markdown('<div class="live-result-card">', unsafe_allow_html=True)
            st.image(image_to_process, caption="Captured image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="live-result-card">', unsafe_allow_html=True)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Recognition result", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if not results:
            st.warning("No valid face detected.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        for r in results:
            if r["status"] == "KNOWN":
                mark_present(r["name"], r["top1"])
                st.success(f"{r['name']} marked present. Score: {r['top1']:.3f}, gap: {r['gap']:.3f}")

            elif r["status"] == "MANUAL_CHECK":
                st.session_state.manual_queue.append(r)
                st.warning(f"Manual verification needed. Top guess: {r['name']} | score: {r['top1']:.3f}, gap: {r['gap']:.3f}")

            else:
                st.error(f"Face could not be confidently identified. Best score: {r['top1']:.3f}, gap: {r['gap']:.3f}")

            with st.expander(f"Face {r['face_num']} details", expanded=True):
                preview_col1, preview_col2, table_col = st.columns([1, 1, 2.5])

                with preview_col1:
                    st.write("Captured face")
                    if os.path.exists(r["crop_path"]):
                        st.image(r["crop_path"], width=170)
                    else:
                        st.info("Captured face image not found.")

                with preview_col2:
                    st.write("Most similar Qdrant face")
                    best_img = resolve_image_path(r.get("best_image_path"))
                    if best_img:
                        st.image(best_img, width=170)
                        st.caption(f"Match: {r['name']} | Score: {r['top1']:.3f}")
                    else:
                        st.info("No Qdrant image found. Make sure image_path/source_image/crop_path is saved in Qdrant payload.")

                with table_col:
                    st.write("Top candidates")
                    cand_df = clean_display_table(
                        r["candidates"],
                        drop_cols=["image_path", "path", "source_image", "crop_path"],
                        round_cols=["cosine_score"],
                    )
                    st.dataframe(cand_df, use_container_width=True, hide_index=True)

                    st.write("Distinct people ranking")
                    distinct_df = clean_display_table(
                        r["distinct"],
                        drop_cols=["image_path", "path", "source_image", "crop_path"],
                        round_cols=["best_cosine"],
                    )
                    st.dataframe(distinct_df, use_container_width=True, hide_index=True)
                    st.caption(f"Recognition event saved to Qdrant collection: {ATTENDANCE_COLLECTION}")

        st.markdown('</div>', unsafe_allow_html=True)

def page_manual_verification():
    page_header(
        "Review queue",
        "Manual Verification",
        "Ambiguous matches appear here when the top score is high but the gap is too small.",
    )

    if not st.session_state.manual_queue:
        st.info("No pending manual verification.")
        return

    for i, item in enumerate(list(st.session_state.manual_queue)):
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 1, 3])
            with c1:
                st.write("Captured face")
                if os.path.exists(item["crop_path"]):
                    st.image(item["crop_path"], width=180)

            with c2:
                st.write("Most similar Qdrant face")
                best_img = resolve_image_path(item.get("best_image_path"))
                if best_img:
                    st.image(best_img, width=180)
                    st.caption(f"Match: {item['name']} | Score: {item['top1']:.3f}")
                else:
                    st.info("No Qdrant image found.")

            with c3:
                st.subheader(f"Top guess: {item['name']}")
                st.write(f"Score: {item['top1']:.3f} | Gap: {item['gap']:.3f}")
                manual_df = clean_display_table(
                    item["candidates"],
                    columns=["candidate_name", "cosine_score"],
                    drop_cols=["image_path", "path", "source_image"],
                    round_cols=["cosine_score"],
                )
                st.dataframe(manual_df, use_container_width=True, hide_index=True)

                options = sorted(set([x.get("candidate_name", "UNKNOWN_NAME") if isinstance(x, dict) else x[0] for x in item["candidates"]])) + ["UNKNOWN"]
                chosen = st.selectbox("Correct person", options, key=f"manual_choice_{i}")

                b1, b2 = st.columns(2)
                with b1:
                    if st.button("Confirm Present", key=f"confirm_{i}"):
                        if chosen != "UNKNOWN":
                            mark_present(chosen, item["top1"])
                        st.session_state.manual_queue.pop(i)
                        st.rerun()
                with b2:
                    if st.button("Remove from Queue", key=f"remove_{i}"):
                        st.session_state.manual_queue.pop(i)
                        st.rerun()


def page_records():
    page_header(
        "Attendance records",
        "Records",
        "View the full attendance log, update students, and download the CSV file.",
    )

    df = load_attendance()

    col1, col2, col3 = st.columns(3)
    with col1:
        stat_card("Present", len(df[df["status"] == "Present"]) if not df.empty else 0, "Currently marked present")
    with col2:
        stat_card("Total records", len(df), "Rows in attendance CSV")
    with col3:
        stat_card("Manual queue", len(st.session_state.manual_queue), "Pending review items")

    st.write("")
    st.dataframe(df, use_container_width=True, hide_index=True)

    present_names = df[df["status"] == "Present"]["name"].tolist() if not df.empty else []
    if present_names:
        st.subheader("Mark Leave")
        c1, c2 = st.columns([2, 1])
        with c1:
            name = st.selectbox("Select student", present_names)
        with c2:
            st.write("")
            if st.button("Mark Left", use_container_width=True):
                mark_left(name)
                st.success(f"{name} marked left.")
                st.rerun()

    st.write("")
    download_col, _ = st.columns([1, 2])
    with download_col:
        if os.path.exists(ATTENDANCE_CSV):
            with open(ATTENDANCE_CSV, "rb") as f:
                st.download_button(
                    "Download Attendance Log",
                    f,
                    file_name=ATTENDANCE_CSV,
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.info("No attendance CSV file found yet.")

    if st.button("Clear attendance log"):
        save_attendance(pd.DataFrame(columns=["name", "status", "entry_time", "leave_time", "last_seen", "best_score"]))
        st.success("Attendance log cleared.")
        st.rerun()


def page_students():
    page_header(
        "Roster",
        "Students",
        "Students currently enrolled in the Qdrant face database.",
    )

    ok, msg = qdrant_ready()
    if not ok:
        st.error(f"Qdrant connection failed: {msg}")
        return

    if st.button("Load students from Qdrant"):
        try:
            people, total_points = list_people_in_qdrant()
            st.success(f"Found {len(people)} unique people and {total_points} total vector points.")
            st.dataframe(pd.DataFrame({"student_name": people}), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not load students: {e}")


def page_enroll():
    page_header(
        "Roster management",
        "Enroll Student",
        "Upload a student picture or take one now, then preprocess and save only this student to Qdrant.",
    )

    with st.container(border=True):
        st.markdown(
            """
            <div class="enroll-step-head">
                <span class="enroll-step-num">1</span>
                <span class="enroll-step-title">Dataset Details</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        person_name = st.text_input(
            "Student folder name",
            placeholder="Example: Fatma",
            key="enroll_person_name",
        )

    st.write("")

    with st.container(border=True):
        st.markdown(
            """
            <div class="enroll-step-head">
                <span class="enroll-step-num">2</span>
                <span class="enroll-step-title">Upload Images</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        upload_tab, camera_tab = st.tabs(["Upload Picture", "Take Picture Now"])

        uploaded_files = []
        camera_photo = None

        with upload_tab:
            uploaded_files = st.file_uploader(
                "Upload face images",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="enroll_upload_files",
            )

        with camera_tab:
            st.caption("Allow camera permission in the browser, then take one student photo.")
            camera_photo = st.camera_input(
                "Take student photo",
                key="enroll_camera_photo",
            )

        st.markdown(
            """
            <div class="enroll-green-note">
                i Best results with clear, front-facing images in good lighting.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Save Images to Dataset", use_container_width=True, key="save_student_images_btn"):
            if not person_name.strip():
                st.error("Enter student folder name first.")
            else:
                safe_person_name = person_name.strip()
                out_dir = MY_DATASET_DIR / safe_person_name
                out_dir.mkdir(parents=True, exist_ok=True)

                saved = 0

                if uploaded_files:
                    for f in uploaded_files:
                        suffix = Path(f.name).suffix.lower() or ".jpg"
                        clean_name = Path(f.name).stem.replace(" ", "_")
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        out_path = out_dir / f"{clean_name}_{ts}{suffix}"
                        out_path.write_bytes(f.getbuffer())
                        saved += 1

                if camera_photo is not None:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    out_path = out_dir / f"camera_{ts}.jpg"
                    out_path.write_bytes(camera_photo.getbuffer())
                    saved += 1

                if saved == 0:
                    st.error("Upload an image or take a picture first.")
                else:
                    st.success(f"Saved {saved} image(s) to {out_dir}")
                    st.info("Next: click Preprocess Only This Student, then Save Only This Student to Qdrant.")

    st.write("")

    with st.container(border=True):
        st.markdown(
            """
            <div class="enroll-step-head">
                <span class="enroll-step-num">3</span>
                <div>
                    <div class="enroll-step-title">Run Pipeline for This Student Only</div>
                    <div class="enroll-step-sub">This does not process everyone again and does not recreate Qdrant.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        replace_old = st.checkbox(
            "Replace old vectors for this student before saving",
            value=False,
            help="Turn this on if you want to update the same student and avoid duplicate old embeddings.",
        )

        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown(
                """
                <div class="enroll-action-card">
                    <div class="enroll-action-title">Preprocess Only This Student</div>
                    <div class="enroll-action-sub">Clean faces and prepare data without saving to Qdrant.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with info_col2:
            st.markdown(
                """
                <div class="enroll-action-card">
                    <div class="enroll-action-title">Save Only This Student to Qdrant</div>
                    <div class="enroll-action-sub">Save processed data directly to Qdrant collection.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        c1, c2 = st.columns(2)

        with c1:
            if st.button("Preprocess Only This Student", use_container_width=True, key="preprocess_one_student_btn"):
                if not person_name.strip():
                    st.error("Enter student folder name first.")
                else:
                    with st.spinner("Preprocessing only this student..."):
                        ok, message, count = preprocess_one_student_to_clean_faces(person_name.strip())
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)

        with c2:
            if st.button("Save Only This Student to Qdrant", use_container_width=True, key="save_one_student_qdrant_btn"):
                if not person_name.strip():
                    st.error("Enter student folder name first.")
                else:
                    ok, msg = qdrant_ready()
                    if not ok:
                        st.error(f"Qdrant connection failed: {msg}")
                    else:
                        with st.spinner("Saving only this student to Qdrant..."):
                            ok2, message, count = save_one_student_to_qdrant(
                                person_name.strip(),
                                replace_old=replace_old,
                            )
                        if ok2:
                            st.success(message)
                            try:
                                people, total_points = list_people_in_qdrant()
                                st.info(
                                    f"Qdrant collection {st.session_state.collection}: "
                                    f"{len(people)} students, {total_points} total vectors."
                                )
                            except Exception as e:
                                st.warning(f"Saved, but could not count Qdrant points: {e}")
                        else:
                            st.error(message)

        st.markdown(
            f"""
            <div class="enroll-folder-note">
                Expected folders: {MY_DATASET_DIR} / student name -> {CLEAN_FACE_DIR} / student name -> Qdrant collection {st.session_state.collection}
            </div>
            """,
            unsafe_allow_html=True,
        )

def page_connections():
    page_header(
        "System",
        "Connections",
        "Configure the Qdrant database connection used by face recognition.",
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Qdrant Connection")
    st.session_state.qdrant_host = st.text_input("Host", st.session_state.qdrant_host)
    st.session_state.qdrant_port = st.number_input("Port", value=int(st.session_state.qdrant_port), step=1)
    st.session_state.collection = st.text_input("Face collection", st.session_state.collection)
    st.text_input("Attendance events collection", ATTENDANCE_COLLECTION, disabled=True)

    if st.button("Test Connection", use_container_width=False):
        ok, msg = qdrant_ready()
        if ok:
            st.success("Qdrant connected successfully.")
            try:
                ensure_attendance_collection(get_qdrant_client())
                st.success(f"Attendance events collection ready: {ATTENDANCE_COLLECTION}")
            except Exception as e:
                st.error(f"Could not create attendance events collection: {e}")
        else:
            st.error(msg)
    st.markdown("</div>", unsafe_allow_html=True)


def page_settings():
    page_header(
        "System",
        "Settings",
        "Manage recognition preferences, attendance files, and quick maintenance actions.",
    )

    qdrant_ok, _ = qdrant_ready()
    csv_exists = os.path.exists(ATTENDANCE_CSV)
    capture_count = len(list(CAPTURE_DIR.glob("*.jpg"))) if CAPTURE_DIR.exists() else 0

    st.markdown('<div class="settings-grid">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="settings-card">
            <div class="settings-title">System Overview</div>
            <div class="settings-sub">Current application status.</div>
            <div class="settings-mini-row"><span>Qdrant</span><span class="settings-badge">{'Online' if qdrant_ok else 'Offline'}</span></div>
            <div class="settings-mini-row"><span>FaceNet Model</span><span class="settings-badge">Ready</span></div>
            <div class="settings-mini-row"><span>Attendance CSV</span><span class="settings-badge">{'Ready' if csv_exists else 'Empty'}</span></div>
            <div class="settings-mini-row"><span>Saved Captures</span><span class="settings-badge">{capture_count}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="settings-card">
            <div class="settings-title">Current Paths</div>
            <div class="settings-sub">Folders used by the attendance system.</div>
            <div class="settings-mini-row"><span>Dataset</span><span>{MY_DATASET_DIR}</span></div>
            <div class="settings-mini-row"><span>Clean Faces</span><span>{CLEAN_FACE_DIR}</span></div>
            <div class="settings-mini-row"><span>Captures</span><span>{CAPTURE_DIR}</span></div>
            <div class="settings-mini-row"><span>CSV Log</span><span>{ATTENDANCE_CSV}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Recognition Defaults")
    st.session_state.threshold = st.slider("Recognition threshold", 0.40, 0.95, float(st.session_state.threshold), 0.01, key="settings_threshold")
    st.session_state.gap = st.slider("Similarity gap", 0.00, 0.25, float(st.session_state.gap), 0.01, key="settings_gap")
    st.session_state.top_k = st.slider("Top K candidates", 3, 20, int(st.session_state.top_k), 1, key="settings_topk")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Maintenance")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear Manual Queue", use_container_width=True):
            st.session_state.manual_queue = []
            st.success("Manual verification queue cleared.")
    with c2:
        if st.button("Refresh Page", use_container_width=True):
            st.rerun()
    st.caption("Use Connections to edit Qdrant host, port, and collection names.")
    st.markdown("</div>", unsafe_allow_html=True)


def page_analytics():
    page_header(
        "Reports",
        "Analytics",
        "Track attendance bars, verification percentage, and recognition summary for your capstone demo.",
    )

    df = load_attendance()

    total_records = len(df)
    present_count = len(df[df["status"] == "Present"]) if not df.empty else 0
    left_count = len(df[df["status"] == "Left"]) if not df.empty else 0
    pending_count = len(st.session_state.manual_queue)
    avg_score = 0.0

    if not df.empty and "best_score" in df.columns:
        scores = pd.to_numeric(df["best_score"], errors="coerce").dropna()
        if len(scores) > 0:
            avg_score = float(scores.mean())

    total_for_pct = max(present_count + left_count + pending_count, 1)
    present_pct = int((present_count / total_for_pct) * 100)
    left_pct = int((left_count / total_for_pct) * 100)
    pending_pct = int((pending_count / total_for_pct) * 100)
    weekly_html = weekly_bars_html(df)
    present_gauge_deg = gauge_degrees(present_pct)

    qdrant_ok, _ = qdrant_ready()
    qdrant_status_pct = 100 if qdrant_ok else 12
    qdrant_status_text = "Online" if qdrant_ok else "Offline"
    csv_status_ok = os.path.exists(ATTENDANCE_CSV)
    csv_status_pct = 100 if csv_status_ok else 12
    csv_status_text = "Ready" if csv_status_ok else "Empty"

    st.markdown(
        f"""
        <div class="analytics-grid">
            <div class="analytics-card"><div class="analytics-label">Total Records</div><div class="analytics-value">{total_records}</div></div>
            <div class="analytics-card"><div class="analytics-label">Present</div><div class="analytics-value">{present_count}</div></div>
            <div class="analytics-card"><div class="analytics-label">Pending</div><div class="analytics-value">{pending_count}</div></div>
            <div class="analytics-card"><div class="analytics-label">Average Score</div><div class="analytics-value">{avg_score:.2f}</div></div>
        </div>
        <div class="dashboard-grid">
            <div class="dash-card">
                <div class="dash-title">Weekly Attendance Bars</div>
                <div class="bar-week">{weekly_html}</div>
            </div>
            <div class="dash-card gauge-card">
                <div class="dash-title">Present Percentage</div>
                <div class="gauge" style="--gauge-deg:{present_gauge_deg};"></div>
                <div class="gauge-value">{present_pct}%</div>
                <div style="font-size:12px;color:#7B8A82;font-weight:800;">Present from active records</div>
                <div class="legend"><span>{present_pct}% Present</span><span>{left_pct}% Left</span><span>{pending_pct}% Pending</span></div>
            </div>
            <div class="dash-card">
                <div class="dash-title">System Status</div>
                <div class="analytics-row"><div class="analytics-status">Qdrant DB</div><div class="analytics-bar-bg"><div class="analytics-bar-fill" style="width:{qdrant_status_pct}%"></div></div><div class="analytics-count">{qdrant_status_text}</div></div>
                <div class="analytics-row"><div class="analytics-status">FaceNet</div><div class="analytics-bar-bg"><div class="analytics-bar-fill" style="width:100%;background:#8EB69B"></div></div><div class="analytics-count">Ready</div></div>
                <div class="analytics-row"><div class="analytics-status">CSV Log</div><div class="analytics-bar-bg"><div class="analytics-bar-fill" style="width:{csv_status_pct}%;background:#DAF1DE"></div></div><div class="analytics-count">{csv_status_text}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    growth_html = attendance_growth_svg(df)
    growth_component = f"""
    <html>
    <head>
    <style>
        body{{margin:0;background:transparent;font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}}
        .growth-chart-card-full{{
            background:white;
            border:1px solid rgba(5,31,32,.08);
            border-radius:22px;
            padding:16px 18px 12px 18px;
            box-shadow:0 8px 20px rgba(5,31,32,.06);
            max-width:620px;
            margin:0 auto;
            box-sizing:border-box;
            overflow:hidden;
        }}
        .chart-pill-title{{
            background:#235347;
            color:white;
            border-radius:999px;
            padding:8px 16px;
            text-align:center;
            font-size:12px;
            font-weight:900;
            margin:0 auto 6px auto;
            max-width:220px;
        }}
        .growth-svg{{width:100%;height:150px;display:block;}}
        .axis-label{{font-size:10px;fill:#6f7d75;font-weight:700;}}
        .grid-line{{stroke:#E6EFE9;stroke-width:1;}}
    </style>
    </head>
    <body>
        <div class="growth-chart-card-full">
            <div class="chart-pill-title">Attendance Growth</div>
            {growth_html}
        </div>
    </body>
    </html>
    """
    components.html(growth_component, height=215, scrolling=False)

    if df.empty:
        st.info("No records yet.")
        return

    st.markdown("<div class='bar-help'>Click any day to update the analytics records below.</div>", unsafe_allow_html=True)
    day_cols = st.columns(7)
    for day_col, day_item in zip(day_cols, ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]):
        with day_col:
            day_count = next((count for d, count, _, _ in weekly_attendance_counts(df) if d == day_item), 0)
            label = f"{day_item} ({day_count})"
            if st.button(label, key=f"analytics_week_bar_{day_item}", use_container_width=True):
                st.session_state.analytics_day = day_item
                st.rerun()

    selected_day = st.session_state.get("analytics_day")
    analytics_df = filter_by_weekday(df, selected_day) if selected_day else df

    st.markdown('<div class="analytics-wrap">', unsafe_allow_html=True)
    if selected_day:
        st.subheader(f"{selected_day} Records")
        if st.button("Show All Days", key="analytics_show_all_days"):
            st.session_state.analytics_day = None
            st.rerun()
    else:
        st.subheader("Recent Records")

    if analytics_df.empty:
        st.info(f"No records found for {selected_day}." if selected_day else "No records yet.")
    else:
        st.dataframe(analytics_df.tail(8), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


dash_view_from_query = st.query_params.get("dash_view")
if dash_view_from_query:
    allowed_views = {"recent", "students", "present", "left", "pending", "day"}
    if dash_view_from_query in allowed_views:
        st.session_state.dashboard_view = dash_view_from_query
        st.session_state.page = "Dashboard"
    st.query_params.clear()

nav_from_query = st.query_params.get("nav")
if nav_from_query:
    st.session_state.page = nav_from_query
    st.query_params.clear()

sidebar()

page = st.session_state.page

if page == "Dashboard":
    page_dashboard()
elif page == "Analytics":
    page_analytics()
elif page == "Live Recognition":
    page_live()
elif page == "Manual Verification":
    page_manual_verification()
elif page == "Records":
    page_records()
elif page == "Students":
    page_students()
elif page == "Enroll Student":
    page_enroll()
elif page == "Connections":
    page_connections()
elif page == "Settings":
    page_settings()
else:
    page_dashboard()
