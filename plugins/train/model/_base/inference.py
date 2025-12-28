#! /usr/env/bin/python3
""" Handles the recompilation of a Faceswap model into a version that can be used for inference """
from __future__ import annotations
import logging
import typing as T

import keras

from lib.logger import parse_class_init
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    import keras.src.ops.node

logger = logging.getLogger(__name__)


class Inference():
    """ Calculates required layers and compiles a saved model for inference.

    Parameters
    ----------
    saved_model: :class:`keras.Model`
        The saved trained Faceswap model
    switch_sides: bool
        ``True`` if the swap should be performed "B" > "A" ``False`` if the swap should be
        "A" > "B"
    """
    def __init__(self, saved_model: keras.Model, switch_sides: bool) -> None:
        logger.debug(parse_class_init(locals()))

        self._layers: list[keras.Layer] = [lyr for lyr in saved_model.layers
                                           if not isinstance(lyr, keras.layers.InputLayer)]
        """list[:class:`keras.layers.Layer]: All the layers that exist within the model excluding
        input layers """

        self._input = self._get_model_input(saved_model, switch_sides)
        """:class:`keras.KerasTensor`: The correct input for the inference model """

        self._name = f"{saved_model.name}_inference"
        """str: The name for the final inference model"""

        self._model = self._build()
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def model(self) -> keras.Model:
        """ :class:`keras.Model`: The Faceswap model, compiled for inference. """
        return self._model

    def _get_model_input(self, model: keras.Model, switch_sides: bool) -> list[keras.KerasTensor]:
        """ Obtain the inputs for the requested swap direction.

        Parameters
        ----------
        saved_model: :class:`keras.Model`
            The saved trained Faceswap model
        switch_sides: bool
            ``True`` if the swap should be performed "B" > "A" ``False`` if the swap should be
            "A" > "B"

        Returns
        -------
        list[]:class:`keras.KerasTensor`]
            The input tensor to feed the model for the requested swap direction
        """
        inputs: list[keras.KerasTensor] = model.input
        assert len(inputs) == 2, "Faceswap models should have exactly 2 inputs"
        idx = 0 if switch_sides else 1
        retval = inputs[idx]
        logger.debug("model inputs: %s, idx: %s, inference_input: '%s'",
                     [(i.name, i.shape[1:]) for i in inputs], idx, retval.name)
        return [retval]

    def _get_candidates(self, input_tensors: list[keras.KerasTensor | keras.Layer]
                        ) -> T.Generator[tuple[keras.Layer, list[keras.src.ops.node.KerasHistory]],
                                         None, None]:
        """ Given a list of input tensors, get all layers from the main model which have the given
        input tensors marked as Inbound nodes for the model

        Parameters
        ----------
        input_tensors: list[:class:`keras.KerasTensor` | :class:`keras.Layer`]
            List of Tensors that act as an input to a layer within the model

        Yields
        ------
        tuple[:class:`keras.KerasLayer`, list[:class:`keras.src.ops.node.KerasHistory']
            Any layer in the main model that use the given input tensors as an input along with the
            corresponding keras inbound history
        """
        unique_input_names = set()
        for i in input_tensors:
            if hasattr(i, "_keras_history"):
                unique_input_names.add(i._keras_history.operation.name)
            else:
                unique_input_names.add(i.name)
        for layer in self._layers:

            history = [tensor._keras_history  # pylint:disable=protected-access
                       for node in layer._inbound_nodes  # pylint:disable=protected-access
                       for parent in node.parent_nodes
                       for tensor in parent.outputs]

            unique_inbound_names = set(h.operation.name for h in history)
            if not unique_input_names.issubset(unique_inbound_names):
                logger.debug("%s: Skipping candidate '%s' unmatched inputs: %s",
                             unique_input_names, layer.name, unique_inbound_names)
                continue

            logger.debug("%s: Yielding candidate '%s'. History: %s",
                         unique_input_names, layer.name, [(h.operation.name, h.node_index)
                                                          for h in history])
            yield layer, history

    @T.overload
    def _group_inputs(self, layer: keras.Layer, inputs: list[tuple[keras.Layer, int]]
                      ) -> list[list[tuple[keras.Layer, int]]]:
        ...

    @T.overload
    def _group_inputs(self, layer: keras.Layer, inputs: list[keras.src.ops.node.KerasHistory]
                      ) -> list[list[keras.src.ops.node.KerasHistory]]:
        ...

    def _group_inputs(self, layer, inputs):
        """ Layers can have more than one input. In these instances we need to group the inputs
        and the layers' inbound nodes to correspond to inputs per instance.

        Parameters
        ----------
        layer: :class:`keras.Layer`
            The current layer being processed
        inputs: list[:class:`keras.KerasTensor`] | list[:class:`keras.src.ops.node.KerasHistory`]
            List of input tensors or inbound keras histories to be grouped per layer input

        Returns
        -------
        list[list[tuple[:class:`keras.Layer`, int]]] |
        list[list[:class:`keras.src.ops.node.KerasHistory`]
            A list of list of input layers  and the corresponding node index or inbound keras
            histories
        """
        layer_inputs = 1 if isinstance(layer.input, keras.KerasTensor) else len(layer.input)
        num_inputs = len(inputs)

        total_calls = num_inputs / layer_inputs
        if not total_calls.is_integer():
            logger.warning("Number of inputs (%s) is not divisible by layer inputs (%s). "
                           "Truncating to nearest integer.", num_inputs, layer_inputs)
        total_calls = int(total_calls)

        retval = [inputs[i * layer_inputs: i * layer_inputs + layer_inputs]
                  for i in range(total_calls)]

        return retval

    def _layers_from_inputs(self,
                            input_tensors: list[keras.KerasTensor | keras.Layer],
                            node_indices: list[int]
                            ) -> tuple[list[keras.Layer],
                                       list[keras.src.ops.node.KerasHistory],
                                       list[int]]:
        """ Given a list of input tensors and their corresponding inbound node ids, return all of
        the layers for the model that uses the given nodes as their input

        Parameters
        ----------
        input_tensors: list[:class:`keras.KerasTensor` | :class:`keras.Layer`]
            List of Tensors that act as an input to a layer within the model
        node_indices: list[int]
            The list of node indices corresponding to the inbound node index of the given layers

        Returns
        -------
        list[:class:`keras.layers.Layer`]
            Any layers from the model that use the given inputs as its input. Empty list if there
            are no matches
        list[:class:`keras.src.ops.node.KerasHistory`]
            The keras inbound history for the layers
        list[int]
            The output node index for the layer, used for the inbound node index of the next layer
        """
        retval: tuple[list[keras.Layer],
                      list[keras.src.ops.node.KerasHistory],
                       list[int]] = ([], [], [])
        for layer, history in self._get_candidates(input_tensors):
            grp_inputs = self._group_inputs(layer, list(zip(input_tensors, node_indices)))
            grp_hist = self._group_inputs(layer, history)

            for input_group in grp_inputs:  # pylint:disable=not-an-iterable
                have = [(i[0].name, i[1]) for i in input_group]
                for out_idx, hist in enumerate(grp_hist):
                    requires = [(h.operation.name, h.node_index) for h in hist]
                    
                    # Sort both to ensure order doesn't matter for valid comparison
                    if sorted(have) != sorted(requires):
                        logger.debug("%s: Skipping '%s'. Requires %s. Output node index: %s",
                                     have, layer.name, requires, out_idx)
                        continue
                    
                    # Handle multi-output layers:
                    # If this layer produces multiple outputs (e.g. fc_both producing [face, mask]),
                    # we must append it *multiple times* to the frontier so the NEXT layer
                    # sees the correct number of input sources.
                    num_outputs = 1
                    # In Keras, layer.output might be a list if multiple outputs
                    # But we need to use the method appropriate for Keras 3 or check the tensor count
                    try:
                        if isinstance(layer.output, list):
                            num_outputs = len(layer.output)
                    except AttributeError:
                        pass # Fallback to 1 if .output is not standard/accessible
                    
                    for _ in range(num_outputs):
                        retval[0].append(layer)
                        retval[1].append(hist)
                        retval[2].append(out_idx)

        logger.debug("Got layers %s for input_tensors: %s",
                     [x.name for x in retval[0]], [t.name for t in input_tensors])
        return retval

    def _build_layers(self,
                      layers: list[keras.Layer],
                      history: list[keras.src.ops.node.KerasHistory],
                      inputs: list[keras.KerasTensor]) -> list[keras.KerasTensor]:
        """ Compile the given layers with the given inputs

        Parameters
        ----------
        layers: list[:class:`keras.Layer`]
            The layers to be called with the given inputs
        history: list[:class:`keras.src.ops.node.KerasHistory`]
            The corresponding keras inbound history for the layers
        inputs: list[:class:`keras.KerasTensor]
            The inputs for the given layers

        Returns
        -------
        list[:class:`keras.KerasTensor`]
            The list of compiled layers
        """
        retval = []
        given_order = [i._keras_history.operation.name  # pylint:disable=protected-access
                       for i in inputs]
        for layer, hist in zip(layers, history):
            layer_input = [inputs[given_order.index(h.operation.name)]
                           for h in hist if h.operation.name in given_order]
            if layer_input != inputs:
                logger.debug("Sorted layer inputs %s to %s",
                             given_order,
                             [i._keras_history.operation.name  # pylint:disable=protected-access
                              for i in layer_input])

            if isinstance(layer_input, list) and len(layer_input) == 1:
                # Flatten single inputs to stop Keras warnings
                actual_input = layer_input[0]
            else:
                actual_input = layer_input

            built = layer(actual_input)
            built = built if isinstance(built, list) else [built]
            logger.debug(
                "Compiled layer '%s' from input(s) %s",
                layer.name,
                [i._keras_history.operation.name  # pylint:disable=protected-access
                 for i in layer_input])
            retval.extend(built)

        logger.debug(
            "Compiled layers %s from input %s",
            [x._keras_history.operation.name for x in retval],  # pylint:disable=protected-access
            [x._keras_history.operation.name for x in inputs])  # pylint:disable=protected-access
        return retval

    def _build(self):
        """ Extract the sub-models from the saved model that are required for inference.

        Returns
        -------
        :class:`keras.Model`
            The model compiled for inference
        """
        logger.debug("Compiling inference model")

        layers = self._input
        node_index = [0]
        built = layers

        while True:
            layers, history, node_index = self._layers_from_inputs(layers, node_index)
            if not layers:
                break

            built = self._build_layers(layers, history, built)

        assert len(self._input) == 1
        if len(built) > 1:
            # Smart selection for multiple outputs (e.g. face + mask)
            print(f"DEBUG: Found {len(built)} model outputs: {[(b.name, b.shape) for b in built]}")
            
            # 1. Prefer output with 3 channels (RGB)
            rgb_outputs = [b for b in built if b.shape[-1] == 3]
            if rgb_outputs:
                 print(f"DEBUG: Selected output by channel count (3): {rgb_outputs[0].name}")
                 built = [rgb_outputs[0]]

            # 2. Prefer output with "face" in name (or user specified override)
            elif any(name_match in b.name.lower() for b in built for name_match in ["face", "keras_tensor_383", "keras_tensor_384"]):
                face_outputs = [b for b in built if any(name_match in b.name.lower() for name_match in ["face", "keras_tensor_383", "keras_tensor_384"])]
                print(f"DEBUG: Selected output by name: {face_outputs[0].name}")
                built = [face_outputs[0]]
            
            # 3. Fallback
            else:
                 logger.warning("Inference model has multiple outputs (%s) and could not identify "
                                "correct face output. Using the first one: %s",
                                len(built), built[0].name)
                 print(f"DEBUG: Fallback selection: {built[0].name}")
                 built = [built[0]]
        
        assert len(built) > 0
        retval = keras.Model(inputs=self._input[0], outputs=built[0], name=self._name)
        logger.debug("Compiled inference model '%s': %s", retval.name, retval)

        return retval


__all__ = get_module_objects(__name__)
