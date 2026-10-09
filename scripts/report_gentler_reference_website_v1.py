"""Package verified live reconstructions without redistributing local photos."""
import hashlib,json,shutil
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/gentler_reference_website_v1'
IDS=['263','6681','10149']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    receipt=json.loads((OUT/'receipt.json').read_text());assert len(receipt)==4
    assert {(r['identity'],r['gentler_details']) for r in receipt}=={('263',False),*[(i,True) for i in IDS]}
    assert all(r['status']=='complete' and r['visible_exact'] and r['raw_exact'] and r['finished_exact'] for r in receipt)
    pack=Path.home()/'Downloads/All Face Datasets/Ready to test sets/Reference reconstruction - 3 people - 2026-10-09'
    instructions='''Website replay - experimental gentler details

Open http://127.0.0.1:8766/ on this laptop.
1. Upload Eyes test/Damaged input.png from this person's set.
2. Upload Eyes test/Mask - white means reconstruct.png.
3. Choose Use 1-4 reference photos. Add Reference photos/Reference 01.png
   through Reference 04.png in their original order.
4. Choose 1024 px, Ignore opaque cover colour ON, Use mask exactly,
   Match colour at edges ON and Gentler eye details ON.
5. Reconstruct. Repeat with Gentler eye details OFF for the preserved control.

Do not upload the evaluation-only original as a reference.
The included image is the actual local API result at seed17, DDIM30, scale0.8.
It exactly matches the corresponding research image. Visible pixels are unchanged.
This setting improved three-person/three-seed mean SSIM, but person10149 lost
mean FaceNet similarity. Iris colour, gaze and eyebrow shape are still imperfect.
SSIM and identity cosine are not correctness percentages.
No training, checkpoint replacement or appearance/texture prototype is deployed.
'''
    for index,i in enumerate(IDS,1):
        row=next(r for r in receipt if r['identity']==i and r['gentler_details'])
        source=Path(row['output']);assert sha(source)==row['sha256']
        folder=ROOT/'outputs/webapp_runs'/row['job_id'];metadata=folder/'metadata.json'
        assert sha(metadata)==row['metadata_sha256']
        local=pack/f'Set 0{index} - Person {i}'/'Website - Gentler details';local.mkdir(exist_ok=True)
        shutil.copy2(source,local/'Eyes - actual website output.png');shutil.copy2(metadata,local/'Run settings.json')
        (local/'READ ME.txt').write_text(instructions)
        control=ROOT/'outputs/focused_reference_finish_v1'/f'{i}_eyes_clean_17.png'
        sheet=Image.new('RGB',(640,355),'white');draw=ImageDraw.Draw(sheet)
        for col,(label,p) in enumerate([('Original setting - CFG5',control),('Optional gentler setting - CFG3',source)]):
            sheet.paste(Image.open(p).convert('RGB').resize((320,320)),(320*col,35));draw.text((320*col+5,10),label,fill='black')
        sheet.save(local/'Control vs website output.png')
    lines=['# Optional gentler-reference app integration','',
           '10 October 2026. Updated app verified at http://127.0.0.1:8766/. The original 8765 process remains available. A request to stop/restart it was rejected by automatic approval review with "blocked by policy"; no forced restart was executed.', '',
           'The new unchecked option changes only CFG 5 to CFG 3 in the fixed 1024 reference mode with opaque-cover removal. Default behavior and the original research generator remain available. It is unavailable for other resolutions, multi-size comparison, reference selection, restoration or evidence-map inference. Unsupported requests are explicitly rejected.', '',
           '## Four actual HTTP/GPU reconstructions','',
           '| Person | Gentler option | Seconds | Raw replay exact | Finished replay exact | Visible pixels exact |',
           '|---|---|---:|---|---|---|']
    for r in receipt:lines.append(f"| {r['identity']} | {'On' if r['gentler_details'] else 'Off'} | {r['seconds']:.2f} | Yes | Yes | Yes |")
    lines+=['','All four match the prior offline outputs byte-for-byte. The app boundary suite passes 22 tests, including scope/type rejection and valid dispatch. Browser checks confirm that the new checkbox starts off, appears in the supported mode, and resets to off when processing detail changes to 512. Old servers do not advertise capability and therefore hide the control.', '',
            'The three actual website outputs, comparison images, run settings and instructions were added under Website - Gentler details in the existing three-person Downloads pack. No original photo or checkpoint was overwritten.', '',
            'For measured quality see FOCUSED_EYE_GUIDANCE_REPLICATION_RESULTS_V2.md. Across three observed people and seeds17/29/43, hole SSIM is 0.414146 to 0.433098, improving 8/9 pairs. Person10149 loses mean FaceNet similarity; iris/gaze/brow errors persist. The option is an explicit comparison, not a universal quality default or unknown-client accuracy claim. These masked SDXL results are not a matched ReF-LDM paper benchmark.', '',
            'The separate appearance branch, guidance/appearance interaction and registered texture transfer are not deployed. Their complete trade-offs and negative results are retained in the respective experiment reports.', '',
            'Receipt: outputs/gentler_reference_website_v1/receipt.json. Browser proof: outputs/gentler_reference_website_v1/browser_controls.jpg. Functional success is distinct from reconstruction fidelity.']
    (ROOT/'research/GENTLER_REFERENCE_WEBSITE_V1.md').write_text('\n'.join(lines)+'\n')
    print('Four replay checks passed; three labelled website test folders created.')

if __name__=='__main__':main()
