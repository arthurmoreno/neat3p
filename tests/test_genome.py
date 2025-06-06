"""Tests creating genomes with various configuration options."""

import os
import sys
import unittest

import neat3p


class TestCreateNew(unittest.TestCase):
    """Tests using unittest."""

    def setUp(self):
        """
        Determine path to configuration file. This path manipulation is
        here so that the script will run successfully regardless of the
        current working directory.
        """
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, "test_configuration")
        self.config = neat3p.Config(
            neat3p.DefaultGenome,
            neat3p.DefaultReproduction,
            neat3p.DefaultSpeciesSet,
            neat3p.DefaultStagnation,
            config_path,
        )
        config2_path = os.path.join(local_dir, "test_configuration2")
        self.config2 = neat3p.Config(
            neat3p.DefaultGenome,
            neat3p.DefaultReproduction,
            neat3p.DefaultSpeciesSet,
            neat3p.DefaultStagnation,
            config2_path,
        )

    def test_unconnected_no_hidden(self):
        """Unconnected network with only input and output nodes."""
        gid = 42
        config = self.config.genome_config
        config.initial_connection = "unconnected"
        config.num_hidden = 0

        g = neat3p.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        g.configure_new(self.config.genome_config)

        print(g)
        self.assertEqual(set(g.nodes), {0})
        assert not g.connections

    def test_unconnected_hidden(self):
        """Unconnected network with hidden nodes."""
        gid = 42
        config = self.config.genome_config
        config.initial_connection = "unconnected"
        config.num_hidden = 2

        g = neat3p.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        g.configure_new(self.config.genome_config)

        print(g)
        self.assertEqual(set(g.nodes), {0, 1, 2})
        assert not g.connections

    def test_fs_neat_no_hidden(self):
        """
        fs_neat with no hidden nodes
        (equivalent to fs_neat_hidden and fs_neat_nohidden with no hidden nodes).
        """
        gid = 42
        config = self.config.genome_config
        config.initial_connection = "fs_neat"
        config.num_hidden = 0

        g = neat3p.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        g.configure_new(config)

        print(g)
        self.assertEqual(set(g.nodes), {0})
        self.assertEqual(len(g.connections), 1)

    def test_fs_neat_hidden_old(self):
        """
        fs_neat (without hidden/nohidden specification) with hidden;
        should output warning about doc/code conflict.
        """
        gid = 42
        config = self.config.genome_config
        config.initial_connection = "fs_neat"
        config.num_hidden = 2

        g = neat3p.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        print("\nThis should output a warning:", file=sys.stderr)
        g.configure_new(config)  # TODO: Test for emitted warning

        print(g)
        self.assertEqual(set(g.nodes), {0, 1, 2})
        self.assertEqual(len(g.connections), 1)

    def test_fs_neat_nohidden(self):
        """fs_neat not connecting hidden nodes."""
        gid = 42
        config = self.config.genome_config
        config.initial_connection = "fs_neat_nohidden"
        config.num_hidden = 2

        g = neat3p.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        g.configure_new(config)

        print(g)
        self.assertEqual(set(g.nodes), {0, 1, 2})
        self.assertEqual(len(g.connections), 1)

    def test_fs_neat_hidden(self):
        """fs_neat with connecting hidden nodes."""
        gid = 42
        config = self.config.genome_config
        config.initial_connection = "fs_neat_hidden"
        config.num_hidden = 2

        g = neat3p.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        g.configure_new(config)

        print(g)
        self.assertEqual(set(g.nodes), {0, 1, 2})
        self.assertEqual(len(g.connections), 3)

    def test_fully_connected_no_hidden(self):
        """
        full with no hidden nodes
        (equivalent to full_nodirect and full_direct with no hidden nodes)
        """
        gid = 42
        config = self.config.genome_config
        config.initial_connection = "full"
        config.num_hidden = 0

        g = neat3p.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        g.configure_new(config)

        print(g)
        self.assertEqual(set(g.nodes), {0})
        self.assertEqual(len(g.connections), 2)

        # Check that each input is connected to the output node
        for i in config.input_keys:
            assert (i, 0) in g.connections

    def test_fully_connected_hidden_nodirect_old(self):
        """
        full (no specification re direct/nodirect) with hidden nodes;
        should output warning re docs/code conflict.
        """
        gid = 42
        config = self.config.genome_config
        config.initial_connection = "full"
        config.num_hidden = 2

        g = neat3p.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        print("\nThis should output a warning:", file=sys.stderr)
        g.configure_new(config)  # TODO: Test for emitted warning

        print(g)
        self.assertEqual(set(g.nodes), {0, 1, 2})
        self.assertEqual(len(g.connections), 6)

        # Check that each input is connected to each hidden node.
        for i in config.input_keys:
            for h in (1, 2):
                assert (i, h) in g.connections

        # Check that each hidden node is connected to the output.
        for h in (1, 2):
            assert (h, 0) in g.connections

        # Check that inputs are not directly connected to the output
        for i in config.input_keys:
            assert (i, 0) not in g.connections

    def test_fully_connected_hidden_nodirect(self):
        """full with no direct input-output connections, only via hidden nodes."""
        gid = 42
        config = self.config.genome_config
        config.initial_connection = "full_nodirect"
        config.num_hidden = 2

        g = neat3p.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        g.configure_new(config)

        print(g)
        self.assertEqual(set(g.nodes), {0, 1, 2})
        self.assertEqual(len(g.connections), 6)

        # Check that each input is connected to each hidden node.
        for i in config.input_keys:
            for h in (1, 2):
                assert (i, h) in g.connections

        # Check that each hidden node is connected to the output.
        for h in (1, 2):
            assert (h, 0) in g.connections

        # Check that inputs are not directly connected to the output
        for i in config.input_keys:
            assert (i, 0) not in g.connections

    def test_fully_connected_hidden_direct(self):
        """full with direct input-output connections (and also via hidden hodes)."""
        gid = 42
        config = self.config.genome_config
        config.initial_connection = "full_direct"
        config.num_hidden = 2

        g = neat3p.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        g.configure_new(config)

        print(g)
        self.assertEqual(set(g.nodes), {0, 1, 2})
        self.assertEqual(len(g.connections), 8)

        # Check that each input is connected to each hidden node.
        for i in config.input_keys:
            for h in (1, 2):
                assert (i, h) in g.connections

        # Check that each hidden node is connected to the output.
        for h in (1, 2):
            assert (h, 0) in g.connections

        # Check that inputs are directly connected to the output
        for i in config.input_keys:
            assert (i, 0) in g.connections

    def test_partially_connected_no_hidden(self):
        """
        partial with no hidden nodes
        (equivalent to partial_nodirect and partial_direct with no hidden nodes)
        """
        gid = 42
        config = self.config2.genome_config
        config.initial_connection = "partial"
        config.connection_fraction = 0.5
        config.num_hidden = 0

        g = neat3p.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        g.configure_new(config)

        print(g)
        self.assertEqual(set(g.nodes), {0})
        self.assertLess(len(g.connections), 2)

    def test_partially_connected_hidden_nodirect_old(self):
        """
        partial (no specification re direct/nodirect) with hidden nodes;
        should output warning re docs/code conflict.
        """
        gid = 42
        config = self.config2.genome_config
        config.initial_connection = "partial"
        config.connection_fraction = 0.5
        config.num_hidden = 2

        g = neat3p.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        print("\nThis should output a warning:", file=sys.stderr)
        g.configure_new(config)  # TODO: Test for emitted warning

        print(g)
        self.assertEqual(set(g.nodes), {0, 1, 2})
        self.assertLess(len(g.connections), 6)

    def test_partially_connected_hidden_nodirect(self):
        """partial with no direct input-output connections, only via hidden nodes."""
        gid = 42
        config = self.config.genome_config
        config.initial_connection = "partial_nodirect"
        config.connection_fraction = 0.5
        config.num_hidden = 2

        g = neat3p.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        g.configure_new(config)

        print(g)
        self.assertEqual(set(g.nodes), {0, 1, 2})
        self.assertLess(len(g.connections), 6)

    def test_partially_connected_hidden_direct(self):
        """
        partial with (potential) direct input-output connections
        (and also, potentially, via hidden hodes).
        """
        gid = 42
        config = self.config.genome_config
        config.initial_connection = "partial_direct"
        config.connection_fraction = 0.5
        config.num_hidden = 2

        g = neat3p.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        g.configure_new(config)

        print(g)
        self.assertEqual(set(g.nodes), {0, 1, 2})
        self.assertLess(len(g.connections), 8)


class TestPruning(unittest.TestCase):
    """Tests using unittest."""

    def setUp(self):
        """
        Determine path to configuration file. This path manipulation is
        here so that the script will run successfully regardless of the
        current working directory.
        """
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, "test_configuration")
        self.config = neat3p.Config(
            neat3p.DefaultGenome,
            neat3p.DefaultReproduction,
            neat3p.DefaultSpeciesSet,
            neat3p.DefaultStagnation,
            config_path,
        )

    def test_empty_network(self):
        gid = 42
        config = self.config.genome_config
        config.initial_connection = "unconnected"
        config.num_hidden = 0

        g = neat3p.DefaultGenome(gid)
        g.configure_new(config)

        g_pruned = g.get_pruned_copy(config)

        self.assertEqual(set(g.nodes.keys()), {0})
        assert not g.connections
        self.assertEqual(set(g.nodes.keys()), set(g_pruned.nodes.keys()))
        self.assertEqual(set(g.connections.keys()), set(g_pruned.connections.keys()))

    def test_nothing_to_prune(self):
        gid = 42
        config = self.config.genome_config
        config.initial_connection = "unconnected"
        config.num_hidden = 0

        g = neat3p.DefaultGenome(gid)
        g.configure_new(config)
        g.add_connection(config, -1, 0, 1.0, True)
        g.add_connection(config, -2, 0, 1.0, True)

        g_pruned = g.get_pruned_copy(config)

        self.assertEqual(set(g.nodes.keys()), set(g_pruned.nodes.keys()))
        self.assertEqual(set(g.connections.keys()), set(g_pruned.connections.keys()))

    def test_unused_node(self):
        gid = 42
        config = self.config.genome_config
        config.initial_connection = "unconnected"
        config.num_hidden = 0

        g = neat3p.DefaultGenome(gid)
        g.configure_new(config)

        new_node_id = config.get_new_node_key(g.nodes)
        ng = g.create_node(config, new_node_id)
        g.nodes[new_node_id] = ng

        g.add_connection(config, -1, 0, 1.0, True)
        g.add_connection(config, -2, 0, 1.0, True)
        g.add_connection(config, -1, new_node_id, 1.0, True)

        g_pruned = g.get_pruned_copy(config)

        self.assertEqual(set(g.nodes.keys()), {0, new_node_id})
        self.assertEqual(set(g.connections.keys()), {(-1, 0), (-2, 0), (-1, new_node_id)})

        self.assertEqual(set(g_pruned.nodes.keys()), {0})
        self.assertEqual(set(g_pruned.connections.keys()), {(-1, 0), (-2, 0)})


if __name__ == "__main__":
    unittest.main()
