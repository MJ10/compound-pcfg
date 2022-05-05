from gflownet import segmenter_controller
import torch
import unittest



class TestControllerMethods(unittest.TestCase):
    device = torch.device("cpu")
    
    def test_sample_backward_actions_merge(self):
        controller = segmenter_controller(device='cpu', args={'nt_states': 5,
                                                              't_states': 5},
                                                        n_vocab=6)
        for i in range(50):
            states = [torch.tensor([0, 0, 6, 1, 0, 6, 1, 2]),
                  torch.tensor([1, 2, 6, 4, 0, 1, 6, 3, 4, 2])]
            original_states = [s.clone() for s in states]
            new_states, B_actions, F_actions, P_B = \
                controller.sample_backward('merge',
                                            torch.zeros(2, 10),
                                            states)
            F_actions = controller.reverse_backward_actions(states, B_actions)
            recovered_states = controller.apply_forward_actions(new_states, F_actions)
            self.assertTrue(len(new_states)==len(states))
            self.assertTrue(len(new_states)==len(B_actions[0]))
            self.assertTrue(len(new_states)==len(B_actions[1]))
            self.assertTrue(len(new_states)==P_B.size(0))
            self.assertTrue((P_B<=0).all())
            self.assertTrue(all([(x==y).all() for x, y in zip(recovered_states, original_states)]))

    def test_sample_backward_actions_untag(self):
        controller = segmenter_controller(device='cpu', args={'nt_states': 5,
                                                              't_states': 5},
                                                        n_vocab=6)
        for i in range(50):
            states = [torch.tensor([0, 3, 7, 0, 2, 9, 1, 2, 10]),
                    torch.tensor([1, 2, 16, 4, 16, 1, 16, 3, 4, 12])]
            original_states = [s.clone() for s in states]
            new_states, B_actions, F_actions, P_B = \
                controller.sample_backward('untag',
                                            torch.zeros(2, 10),
                                            states)
            recovered_states = controller.apply_forward_actions(new_states, F_actions)
            #import pdb; pdb.set_trace()
            self.assertTrue(len(new_states)==len(states))
            self.assertTrue(len(new_states)==len(B_actions[0]))
            self.assertTrue(len(new_states)==len(B_actions[1]))
            self.assertTrue(len(new_states)==P_B.size(0))
            self.assertTrue((P_B<=0).all())
            self.assertTrue(all([(x==y).all() for x, y in zip(recovered_states, original_states)]))
    
    def test_sample_forward_actions_split(self):
        controller = segmenter_controller(device='cpu', args={'nt_states': 5,
                                                              't_states': 5},
                                                        n_vocab=6)
        for i in range(50):
            states = [torch.tensor([0, 1, 0, 6, 1, 2, 1]),
                    torch.tensor([1, 2, 1, 4, 1, 1, 6, 3, 4, 2])]
            original_states = [s.clone() for s in states]
            new_states, F_actions, B_actions, P_F = \
                controller.sample_forward('split',
                                            torch.zeros(2, 10),
                                            states)
            recovered_states = controller.apply_backward_actions(new_states, B_actions)
            #import pdb; pdb.set_trace()
            self.assertTrue(len(new_states)==len(states))
            self.assertTrue(len(new_states)==len(B_actions[0]))
            self.assertTrue(len(new_states)==len(B_actions[1]))
            self.assertTrue(len(new_states)==P_F.size(0))
            self.assertTrue((P_F<=0).all())
            self.assertTrue(all([(x==y).all() for x, y in zip(recovered_states, original_states)]))

    def test_sample_forward_actions_tag(self):
        controller = segmenter_controller(device='cpu', args={'nt_states': 5,
                                                              't_states': 5},
                                                        n_vocab=6)
        for i in range(50):
            states = [torch.tensor([0, 6, 0, 6, 1, 2, 1, 6]),
                    torch.tensor([1, 2, 6, 4, 6, 1, 6, 3, 4, 6])]
            original_states = [s.clone() for s in states]
            new_states, F_actions, B_actions, P_F = \
                controller.sample_forward('tag',
                                            torch.zeros(2, 10, 6),
                                            states)
            recovered_states = controller.apply_backward_actions(new_states, B_actions)
            #import pdb; pdb.set_trace()
            self.assertTrue(len(new_states)==len(states))
            self.assertTrue(len(new_states)==len(B_actions[0]))
            self.assertTrue(len(new_states)==len(B_actions[1]))
            self.assertTrue(len(new_states)==P_F.size(0))
            self.assertTrue((P_F<=0).all())
            self.assertTrue(all([(x==y).all() for x, y in zip(recovered_states, original_states)]))

if __name__ == "__main__":
    unittest.main()