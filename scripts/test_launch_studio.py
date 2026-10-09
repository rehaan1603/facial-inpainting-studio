"""Verify stale-server avoidance without interrupting jobs or launching models."""
import io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
import launch_studio as launch


class LauncherTests(unittest.TestCase):
    def session(self,value):return io.BytesIO(json.dumps(value).encode('utf-8'))

    def test_accepts_matching_revision_even_while_busy(self):
        data=dict(token='local',busy=True,gentler_details_available=True,studio_revision='current')
        with patch.object(launch,'urlopen',return_value=self.session(data)):
            self.assertTrue(launch.ready(launch.studio_url(8765),'current'))

    def test_rejects_stale_or_malformed_sessions(self):
        valid=dict(token='local',busy=False,gentler_details_available=True,studio_revision='current')
        for data in [[],None,{**valid,'studio_revision':'old'},
                     {**valid,'gentler_details_available':'true'},
                     {**valid,'gentler_details_available':1},
                     {**valid,'busy':0},dict(token='legacy',busy=False)]:
            with self.subTest(data=data),patch.object(launch,'urlopen',return_value=self.session(data)):
                self.assertFalse(launch.ready(launch.studio_url(8765),'current'))

    def test_connection_failure_is_not_reused(self):
        with patch.object(launch,'urlopen',side_effect=OSError('unavailable')):
            self.assertFalse(launch.ready(launch.studio_url(8765),'current'))

    def test_reuses_current_server_on_fallback_port(self):
        with patch.object(launch,'listening',side_effect=lambda port:port==8767), \
             patch.object(launch,'ready',side_effect=lambda url,revision:url==launch.studio_url(8767)):
            self.assertEqual(launch.current_server('current'),launch.studio_url(8767))

    def test_reuses_updated_8766_instead_of_stale_8765(self):
        with patch.object(sys,'argv',['launcher']),patch.object(launch,'code_revision',return_value='current'), \
             patch.object(launch,'ready',side_effect=lambda url,revision:url==launch.studio_url(8766)), \
             patch.object(launch.subprocess,'Popen') as process,patch.object(launch.webbrowser,'open') as browser, \
             patch('builtins.print'):
            launch.main();process.assert_not_called();browser.assert_called_once_with(launch.studio_url(8766))

    def test_prefers_current_8765_and_no_browser_mode(self):
        with patch.object(sys,'argv',['launcher','--no-browser']),patch.object(launch,'code_revision',return_value='current'), \
             patch.object(launch,'ready',return_value=True),patch.object(launch.subprocess,'Popen') as process, \
             patch.object(launch.webbrowser,'open') as browser,patch('builtins.print'):
            launch.main();process.assert_not_called();browser.assert_not_called()

    def test_new_server_receives_selected_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'configs').mkdir();(root/'configs/local.json').write_text('{}')
            with patch.object(sys,'argv',['launcher']),patch.object(launch,'ROOT',root), \
                 patch.object(launch,'code_revision',return_value='current'), \
                 patch.object(launch,'listening',return_value=False), \
                 patch.object(launch,'ready',side_effect=[False,False,True]), \
                 patch.object(launch,'available_port',return_value=8767), \
                 patch.object(launch.subprocess,'Popen') as process,patch.object(launch.webbrowser,'open') as browser, \
                 patch('builtins.print') as output:
                launch.main()
                self.assertEqual(process.call_args.args[0][-2:],['--port','8767'])
                browser.assert_called_once_with(launch.studio_url(8767))
                self.assertIn(launch.studio_url(8767),output.call_args_list[0].args[0])

    def test_startup_failure_reports_log_and_does_not_open_browser(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'configs').mkdir();(root/'configs/local.json').write_text('{}')
            with patch.object(sys,'argv',['launcher']),patch.object(launch,'ROOT',root), \
                 patch.object(launch,'code_revision',return_value='current'), \
                 patch.object(launch,'listening',return_value=False), \
                 patch.object(launch,'ready',return_value=False),patch.object(launch,'available_port',return_value=8767), \
                 patch.object(launch.subprocess,'Popen',return_value=MagicMock(poll=lambda:1)), \
                 patch.object(launch.webbrowser,'open') as browser:
                with self.assertRaisesRegex(SystemExit,'studio_server.log'):launch.main()
                browser.assert_not_called()


if __name__=='__main__':unittest.main()
