const assert = require('node:assert/strict');
const {frameGeometry, maskGeometry} = require('../webapp/dist/framing.js');
assert.deepEqual(frameGeometry(800,1200,'crop'),[0,200,800,800,0,0,512,512]);
assert.deepEqual(maskGeometry(800,1200,800,1200,'crop'),frameGeometry(800,1200,'crop'));
assert.deepEqual(frameGeometry(1200,600,'fit'),[0,0,1200,600,0,128,512,256]);
assert.deepEqual(maskGeometry(1200,600,1200,600,'fit'),frameGeometry(1200,600,'fit'));
assert.deepEqual(maskGeometry(512,512,800,1200,'crop'),[0,0,512,512,0,0,512,512]);
assert.throws(()=>maskGeometry(300,200,800,1200,'crop'),/matching/);
console.log('Mask framing: crop, fit, framed masks and mismatched-size rejection passed');
