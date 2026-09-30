"""Regression coverage for native-model terminal allocation."""

import multiconductor as mc
from multiconductor.pycci.model import DottedDict, NUM_PHASES, initialize_terminal_lookup


def test_asymmetric_generators_allocate_one_terminal_block():
    net = mc.create_empty_network(sn_mva=1.0)
    bus = mc.create_bus(net, 20.0)
    mc.create_asymmetric_gen(
        net,
        bus,
        from_phase=(1, 2, 3),
        to_phase=(0, 0, 0),
        p_mw=(0.1, 0.1, 0.1),
        vm_pu=(1.0, 1.0, 1.0),
    )

    net["model"] = DottedDict()
    initialize_terminal_lookup(net)

    generator_count = int(net.asymmetric_gen.index.get_level_values(0).max()) + 1
    expected_terminal_count = (
        net.model.gen_index_start + generator_count * NUM_PHASES * 2
    )
    assert net.model.num_terminals == expected_terminal_count
