"""Start or reuse the local studio independently of the calling terminal."""
import argparse,json,os,subprocess,sys,time,webbrowser
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
URL='http://127.0.0.1:8765/'

def ready():
    try:
        with urlopen(URL+'api/session',timeout=2) as response:
            data=json.load(response)
        return isinstance(data.get('token'),str) and isinstance(data.get('busy'),bool)
    except (OSError,ValueError):return False

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--no-browser',action='store_true');args=parser.parse_args()
    if not ready():
        if not (ROOT/'configs/local.json').is_file():raise SystemExit('Missing configs/local.json. Restore your dataset/model paths using configs/local.example.json.')
        out=ROOT/'outputs';out.mkdir(exist_ok=True)
        options={'cwd':ROOT,'stdin':subprocess.DEVNULL,'close_fds':True}
        if os.name=='nt':options['creationflags']=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW
        else:options['start_new_session']=True
        with (out/'studio_server.log').open('ab',buffering=0) as log:
            process=subprocess.Popen([sys.executable,str(ROOT/'webapp/server.py')],stdout=log,stderr=log,**options)
        for _ in range(60):
            if ready():break
            if process.poll() is not None:raise SystemExit('The studio could not start. See outputs/studio_server.log.')
            time.sleep(.5)
        else:raise SystemExit('Startup is taking longer than expected. See outputs/studio_server.log.')
    print('Inpainting Studio is ready: '+URL)
    print('The local server keeps running after this launcher closes.')
    if not args.no_browser:webbrowser.open(URL)

if __name__=='__main__':main()
