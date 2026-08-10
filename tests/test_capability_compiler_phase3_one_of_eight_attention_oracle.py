import torch
from abi.capability_compiler_phase3_one_of_eight_attention_oracle import one_of_eight

def test_one_of_eight_retains_exactly_one_per_group_with_stable_ties():
    weight=torch.tensor([[3.,3.,1.,0.,0.,0.,0.,0.,-1.,-2.,-4.,0.,0.,0.,0.,0.]])
    retained,energy=one_of_eight(weight)
    assert torch.nonzero(retained[0]).squeeze(1).tolist()==[0,10]
    assert retained[0,0]==3 and retained[0,10]==-4 and 0<energy<1

def test_one_of_eight_rejects_unaligned_input_width():
    try: one_of_eight(torch.ones(2,7))
    except Exception as error: assert "input width" in str(error)
    else: raise AssertionError("unaligned width accepted")
