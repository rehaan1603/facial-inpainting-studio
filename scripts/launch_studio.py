"""Start or reuse the local studio independently of the calling terminal."""
import argparse,json,os,socket,subprocess,sys,time,webbrowser
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from webapp.version import code_revision
PORTS=(8765,8766)

def studio_url(port):return f'http://127.0.0.1:{port}/'

def listening(port):
    try:
        with socket.create_connection(('127.0.0.1',port),timeout=.08):return True
    except OSError:return False

def current_server(revision):
    ports=[*PORTS,*range(8767,8797)]
    for port in ports:
        if port not in PORTS and not listening(port):continue
        url=studio_url(port)
        if ready(url,revision):return url
    return None

def ready(url,revision):
    try:
        with urlopen(url+'api/session',timeout=2) as response:
            data=json.load(response)
        return (isinstance(data,dict) and isinstance(data.get('token'),str)
                and isinstance(data.get('busy'),bool)
                and data.get('gentler_details_available') is True
                and data.get('studio_revision')==revision)
    except (OSError,ValueError):return False

def available_port():
    for port in range(8765,8797):
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
            try:probe.bind(('127.0.0.1',port))
            except OSError:continue
            return port
    raise SystemExit('No available local studio port between 8765 and 8796.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--no-browser',action='store_true');args=parser.parse_args()
    revision=code_revision()
    url=current_server(revision)
    if url is None:
        if not (ROOT/'configs/local.json').is_file():raise SystemExit('Missing configs/local.json. Restore your dataset/model paths using configs/local.example.json.')
        port=available_port();url=studio_url(port)
        out=ROOT/'outputs';out.mkdir(exist_ok=True)
        options={'cwd':ROOT,'stdin':subprocess.DEVNULL,'close_fds':True}
        if os.name=='nt':options['creationflags']=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW
        else:options['start_new_session']=True
        with (out/'studio_server.log').open('ab',buffering=0) as log:
            process=subprocess.Popen([sys.executable,str(ROOT/'webapp/server.py'),'--port',str(port)],stdout=log,stderr=log,**options)
        for _ in range(60):
            if ready(url,revision):break
            if process.poll() is not None:raise SystemExit('The studio could not start. See outputs/studio_server.log.')
            time.sleep(.5)
        else:raise SystemExit('Startup is taking longer than expected. See outputs/studio_server.log.')
    print('Inpainting Studio is ready: '+url)
    print('The local server keeps running after this launcher closes.')
    if not args.no_browser:webbrowser.open(url)

if __name__=='__main__':main()
