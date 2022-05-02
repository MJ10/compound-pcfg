from gflownet import segmenter_controller
import torch
import unittest



class TestControllerMethods(unittest.TestCase):
    device = torch.device("cpu")
    
    def test_sample_backward_actions_merge(self):
        controller = segmenter_controller(device='cpu', args={'pad_sym': 5,
                                                              'split_sym': 6,
                                                              'eos_sym': 10})
        for i in range(50):
            states = [torch.tensor([0, 6, 0, 6, 1, 10]),
                  torch.tensor([1, 2, 6, 4, 6, 1, 6, 3, 4, 10])]
            original_states = [s.clone() for s in states]
            new_states, B_actions, P_B = \
                controller.sample_backward('merge',
                                            torch.zeros(2, 10),
                                            states)
            F_actions = controller.reverse_backward_actions(states, B_actions)
            recovered_states = controller.apply_forward_actions(new_states, F_actions)
            self.assertTrue(len(new_states)==len(states))
            self.assertTrue(len(new_states)==len(B_actions[0]))
            self.assertTrue(len(new_states)==len(B_actions[1]))
            self.assertTrue(len(new_states)==P_B.size(0))
            self.assertTrue(all([(x==y).all() for x, y in zip(recovered_states, original_states)]))
            
if __name__ == "__main__":
    unittest.main()