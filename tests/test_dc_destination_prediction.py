#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `dc_destination_prediction` package."""


import unittest
from click.testing import CliRunner

from dc_destination_prediction import dc_destination_prediction
from dc_destination_prediction import cli


class TestDc_destination_prediction(unittest.TestCase):
    """Tests for `dc_destination_prediction` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'dc_destination_prediction.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
