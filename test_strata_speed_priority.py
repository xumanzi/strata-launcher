import unittest
from strata_optimization import apply_runtime_optimization, model_settings


class SpeedPriorityTests(unittest.TestCase):
    def test_qwen_removes_old_tuning_and_overrides_process_ram_default(self):
        cfg={'backend':'cuda','model_name':'Qwen3.8-Flash-Next-IQ3_S',
             'args':['--native','M.gguf','--max-context','262144','--adapt-swaps','160'],
             'user_args':['--ram-keep-gib=8','--adapt-every','1'],
             'env':{'STRATA_RAM_KEEP_GIB':'8'}}
        result,_=apply_runtime_optimization(cfg,{'mode':'speed'},'model.json',None)
        for key in ('args','user_args'):
            self.assertNotIn('--adapt-swaps',result[key])
            self.assertNotIn('--adapt-every',result[key])
            self.assertNotIn('--ram-keep-gib=8',result[key])
            self.assertEqual(result[key][result[key].index('--ram-keep-gib')+1],'2')
        self.assertNotIn('STRATA_RAM_KEEP_GIB',result['env'])
        self.assertNotIn('--parallel',result['args'])
        self.assertEqual(result['args'][result['args'].index('--max-context')+1],'262144')
        self.assertIn('--adapt-swaps',cfg['args'])
        self.assertEqual(result['args_written'],result['args'])

    def test_inkling_fixes_kv_offload_without_removing_auxiliaries(self):
        cfg={'arch':'inkling2_loop','launcher_speed_priority':True,
             'args':['--native','M.gguf','--max-context','262144','--dspark','D.gguf',
                     '--vision','V.strata','--kv-offload','on','--vram-experts-gb','9.68']}
        result,_=apply_runtime_optimization(cfg,{},'model.json',None)
        self.assertEqual(result['args'][result['args'].index('--kv-offload')+1],'auto')
        self.assertEqual(result['args'][result['args'].index('--vram-reserve-mib')+1],'2048')
        self.assertNotIn('--vram-experts-gb',result['args'])
        self.assertIn('D.gguf',result['args'])
        self.assertIn('V.strata',result['args'])

    def test_speed_keeps_larger_inline_reserve(self):
        cfg={'arch':'inkling2_loop','args':['--native','M.gguf'],
             'user_args':['--vram-reserve-mib=4096']}
        result,_=apply_runtime_optimization(cfg,{'mode':'speed'},'model.json',None)
        self.assertEqual(result['args'][result['args'].index('--vram-reserve-mib')+1],'4096')

    def test_none_effort_is_explicit_and_supported(self):
        self.assertEqual(model_settings({'effort':'none'}),{'reasoning_effort':'none'})


if __name__=='__main__':unittest.main()
