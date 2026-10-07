import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('h3_config', Path(__file__).parents[1] / 'config.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class ConfigTests(unittest.TestCase):
    def cfg(self, **kwargs): return m.parse_config(kwargs)
    def test_presets_and_grid(self):
        for aspect in m.ASPECTS:
            for resolution in m.SHORT_EDGES:
                w,h,n=m.geometry(self.cfg(aspect=aspect,resolution=resolution,seconds=5))
                self.assertEqual(w%32,0);self.assertEqual(h%32,0);self.assertLessEqual(w*h,1344*768)
                self.assertEqual(n,124)
    def test_durations(self):
        for seconds in (.2,1,5,6,8,10,15,30):
            _,_,n=m.geometry(self.cfg(seconds=seconds))
            self.assertEqual(n%17,5);self.assertGreaterEqual(n/24,seconds)
    def test_auto_and_manual_modes(self):
        c=self.cfg()
        self.assertEqual(m.route_mode(c,0,0,0),'text')
        for counts in ((1,0,0),(0,1,0),(0,0,1),(9,3,3)):
            self.assertEqual(m.route_mode(c,*counts),'reference')
        self.assertEqual(m.route_mode(self.cfg(mode='text'),1,0,0),'text')
        with self.assertRaises(ValueError):m.route_mode(self.cfg(mode='first_frame'),2,0,0)
        self.assertEqual(m.route_mode(self.cfg(mode='first_last'),2,0,0),'first_last')
    def test_unlimited_loras_serialization(self):
        l=[{'name':f'x{i}.safetensors','weight':.4,'enabled':True} for i in range(80)]
        c=self.cfg(loras=l);self.assertEqual(len(c['loras']),80)
        self.assertEqual(m.parse_config(json.dumps(c))['loras'],l)
    def test_bad_numbers(self):
        for k,v in [('seconds',float('nan')),('width',-2),('steps',0),('rife_multiplier',1),('llm_tokens',100000)]:
            with self.assertRaises(ValueError):self.cfg(**{k:v,'resolution':'custom'})
        with self.assertRaises(ValueError):self.cfg(resolution='custom',width=777)
    def test_profile(self):
        self.assertEqual(m.effective_sampling(self.cfg()),(4,'euler','simple'))
        self.assertEqual(m.effective_sampling(self.cfg(profile='standard')),(20,'res_multistep','simple'))
        self.assertEqual(m.effective_sampling(self.cfg(profile='custom',steps=6,sampler='er_sde',scheduler='beta')),(6,'er_sde','beta'))
        self.assertEqual(m.effective_sampling(self.cfg(version=2,profile='DMAD 4-step',steps=7,sampler='er_sde')),(7,'er_sde','simple'))
    def test_resolution_selector_formula(self):
        import math
        for aspect in m.ASPECTS:
            for mp in (.1,.7,1.,2.):
                a,b=m.ASPECTS[aspect];scale=math.sqrt(mp*1024*1024/(a*b))
                w,h,_=m.geometry(self.cfg(version=2,resolution='megapixels',megapixels=mp,aspect=aspect))
                self.assertEqual((w,h),(round(a*scale/32)*32,round(b*scale/32)*32))
    def test_migration_and_audio_slots(self):
        c=self.cfg(version=1,profile='DMAD 4-step',dmad_lora='dmad.safetensors')
        self.assertEqual(c['version'],2);self.assertEqual(c['loras'][0]['weight'],1.)
        self.assertEqual(m.parse_config(c)['loras'],c['loras'])
        with self.assertRaises(ValueError):self.cfg(media=[{'kind':'audio','path':f'{i}.wav'} for i in range(3)])
        with self.assertRaises(ValueError):self.cfg(media=[{'kind':'image','path':'a.png','slot':0},{'kind':'image','path':'b.png','slot':0}])
    def test_invalid_media(self):
        for media in ([{'kind':'image','path':'a.exe'}], [{'kind':'audio','path':'a.wav','start':5,'end':2}],
                      [{'kind':'video','path':f'{i}.mp4'} for i in range(4)]):
            with self.assertRaises(ValueError):self.cfg(media=media)
    def test_filenames(self):
        for filename in ('../test','CON','NUL.txt','bad:name',''):
            with self.assertRaises(ValueError):self.cfg(filename=filename)
    def test_save_scope(self):
        with self.assertRaises(ValueError):m.output_directory(self.cfg(output_dir='../outside'),Path.cwd())
        self.assertEqual(m.output_directory(self.cfg(output_dir='H3'),Path.cwd()),(Path.cwd()/'H3').resolve())

if __name__=='__main__':unittest.main(verbosity=2)
