"""Enhanced Gradio UI with progressive dataset filtering."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import gradio as gr
import pandas as pd

from smolagents import GradioUI
from smolagents.models import InferenceClientModel, LiteLLMModel, OpenAIServerModel

from agents_config import create_manager_agent
from data.loader import PHMDataset


class EnhancedGradioUI(GradioUI):
    """Custom UI providing dataset and knowledge base utilities."""

    def __init__(self, agent, kb_dir: str = "knowledge_base", config=None, **kwargs) -> None:
        """Create the UI wrapper.

        Parameters
        ----------
        agent:
            The root agent handling user queries.
        kb_dir:
            Directory containing knowledge base documents.
        """
        super().__init__(agent, **kwargs)
        self.kb_dir = Path(kb_dir)
        self.config = config
        self.metadata_path: Optional[str] = None
        self.data_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------
    def _build_model(self, provider: str, model_id: str, api_key: str):
        """Return a model instance for the selected provider."""
        if provider == "LiteLLM":
            return LiteLLMModel(model_id=model_id, api_key=api_key)
        if provider == "OpenAI":
            return OpenAIServerModel(model_id=model_id, api_key=api_key)
        if provider == "HuggingFace":
            return InferenceClientModel(model_id=model_id, token=api_key)
        raise gr.Error(f"Unsupported provider: {provider}")

    def init_model(self, provider, model_id, api_key, session_state):
        """Initialise the LLM model and store the agent in state."""
        if not api_key:
            raise gr.Error("API key required")
        model = self._build_model(provider, model_id, api_key)
        session_state["agent"] = create_manager_agent(model, self.config)
        return "Model initialized"

    # ------------------------------------------------------------------
    # Dataset handlers
    # ------------------------------------------------------------------
    def load_dataset(self, metadata_path: str, data_path: str):
        """Load dataset and return initial metadata and dropdown choices."""
        if not metadata_path or not data_path:
            raise gr.Error("Specify metadata and data paths")
        ds = PHMDataset(metadata_path, data_path)
        self.metadata_path = metadata_path
        self.data_path = data_path
        df = ds.metadata
        choices = {
            "Dataset_id": sorted(df["Dataset_id"].dropna().unique().tolist()),
            "TYPE": sorted(df["TYPE"].dropna().unique().tolist()),
            "Fault_level": sorted(df["Fault_level"].dropna().unique().tolist()),
            "Domain_id": sorted(df["Domain_id"].dropna().unique().tolist()),
            "Visiable": sorted(df["Visiable"].dropna().unique().tolist()),
            "Id": df["Id"].astype(str).tolist(),
        }
        return (
            df,
            ds,
            gr.Dropdown.update(choices=choices["Dataset_id"], value=None, multiselect=True),
            gr.Dropdown.update(choices=choices["TYPE"], value=None),
            gr.Dropdown.update(choices=choices["Fault_level"], value=None),
            gr.Dropdown.update(choices=choices["Domain_id"], value=None),
            gr.Dropdown.update(choices=choices["Visiable"], value=None),
            gr.Dropdown.update(choices=choices["Id"], value=None),
        )

    def _apply_filters(
        self,
        dataset_ids: List[str] | None,
        type_val: str | None,
        fault_val: str | None,
        domain_val: str | None,
        visible_val: str | None,
        dataset: PHMDataset,
    ):
        """Filter dataset metadata and compute updated dropdown choices."""
        df = dataset.metadata
        filtered = df
        if dataset_ids:
            filtered = filtered[filtered["Dataset_id"].isin(dataset_ids)]
        if type_val:
            filtered = filtered[filtered["TYPE"] == type_val]
        if fault_val:
            filtered = filtered[filtered["Fault_level"] == fault_val]
        if domain_val:
            filtered = filtered[filtered["Domain_id"] == domain_val]
        if visible_val:
            filtered = filtered[filtered["Visiable"] == visible_val]

        def _choices(column: str):
            return sorted(filtered[column].dropna().unique().tolist())

        return (
            filtered,
            gr.Dropdown.update(choices=_choices("TYPE"), value=type_val if type_val in _choices("TYPE") else None),
            gr.Dropdown.update(choices=_choices("Fault_level"), value=fault_val if fault_val in _choices("Fault_level") else None),
            gr.Dropdown.update(choices=_choices("Domain_id"), value=domain_val if domain_val in _choices("Domain_id") else None),
            gr.Dropdown.update(choices=_choices("Visiable"), value=visible_val if visible_val in _choices("Visiable") else None),
            gr.Dropdown.update(choices=filtered["Id"].astype(str).tolist(), value=None),
        )

    def reset_filters(self, dataset: PHMDataset):
        """Reset filtering dropdowns to their initial state."""
        df = dataset.metadata
        return self.load_dataset(str(self.metadata_path), str(self.data_path))[:8]

    def plot_signal(self, selected_id, dataset):
        """Return a plot of the selected signal."""
        if dataset is None or selected_id is None:
            raise gr.Error("Dataset not ready")
        data = dataset.load(int(selected_id))
        fig, ax = plt.subplots()
        ax.plot(data)
        ax.set_title(f"Signal {selected_id}")
        return fig

    def process_split(self, train_ids, val_ids, test_ids, dataset, session_state):
        """Run the PHM agent on a train/val/test split."""
        if dataset is None:
            raise gr.Error("Dataset not ready")
        if "agent" not in session_state:
            raise gr.Error("Initialize model first")
        prompt = (
            f"Use dataset with metadata at {dataset.metadata_path} and data at {dataset.data_path}.\n"
            f"Training IDs: {train_ids}\nValidation IDs: {val_ids}\nTest IDs: {test_ids}"
        )
        result = session_state["agent"].run(prompt)
        return str(result.final_answer)

    # ------------------------------------------------------------------
    # Knowledge base loader
    # ------------------------------------------------------------------
    def load_kb(self, filename):
        """Return the contents of a knowledge base document."""
        if not filename:
            return ""
        path = self.kb_dir / filename
        if not path.exists():
            return f"File {filename} not found"
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------
    def create_chat_tab(self, session_state):
        """Build the interactive chat tab."""
        with gr.Blocks() as chat_demo:
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])

            with gr.Sidebar():
                gr.Markdown(f"# {self.name.replace('_', ' ').capitalize()}\n> Chat with the PHM agent")

                with gr.Accordion("Model Configuration", open=False):
                    provider = gr.Dropdown(["LiteLLM", "OpenAI", "HuggingFace"], label="Provider", value="LiteLLM")
                    model_id = gr.Textbox("gemini/gemini-pro", label="Model ID")
                    api_key = gr.Textbox(label="API Key", type="password")
                    init_btn = gr.Button("Initialize Model")
                    status = gr.Textbox(label="Status", interactive=False)
                    init_btn.click(self.init_model, [provider, model_id, api_key, session_state], [status])

                with gr.Accordion("Dataset", open=False):
                    meta_path = gr.Textbox(label="Metadata Path")
                    data_path = gr.Textbox(label="Data Path")
                    load_btn = gr.Button("Load")
                    metadata_table = gr.DataFrame(label="Metadata")
                    dataset_state = gr.State(None)

                    dataset_id = gr.Dropdown(label="Dataset_id", multiselect=True)
                    type_dd = gr.Dropdown(label="TYPE")
                    fault_dd = gr.Dropdown(label="Fault_level")
                    domain_dd = gr.Dropdown(label="Domain_id")
                    visiable_dd = gr.Dropdown(label="Visiable")
                    reset_btn = gr.Button("Reset Filters")

                    id_select = gr.Dropdown(label="Select ID")
                    plot_output = gr.Plot()

                    load_btn.click(
                        self.load_dataset,
                        [meta_path, data_path],
                        [metadata_table, dataset_state, dataset_id, type_dd, fault_dd, domain_dd, visiable_dd, id_select],
                    )
                    dataset_id.change(
                        self._apply_filters,
                        [dataset_id, type_dd, fault_dd, domain_dd, visiable_dd, dataset_state],
                        [metadata_table, type_dd, fault_dd, domain_dd, visiable_dd, id_select],
                    )
                    type_dd.change(
                        self._apply_filters,
                        [dataset_id, type_dd, fault_dd, domain_dd, visiable_dd, dataset_state],
                        [metadata_table, type_dd, fault_dd, domain_dd, visiable_dd, id_select],
                    )
                    fault_dd.change(
                        self._apply_filters,
                        [dataset_id, type_dd, fault_dd, domain_dd, visiable_dd, dataset_state],
                        [metadata_table, type_dd, fault_dd, domain_dd, visiable_dd, id_select],
                    )
                    domain_dd.change(
                        self._apply_filters,
                        [dataset_id, type_dd, fault_dd, domain_dd, visiable_dd, dataset_state],
                        [metadata_table, type_dd, fault_dd, domain_dd, visiable_dd, id_select],
                    )
                    visiable_dd.change(
                        self._apply_filters,
                        [dataset_id, type_dd, fault_dd, domain_dd, visiable_dd, dataset_state],
                        [metadata_table, type_dd, fault_dd, domain_dd, visiable_dd, id_select],
                    )
                    reset_btn.click(
                        self.reset_filters,
                        [dataset_state],
                        [metadata_table, dataset_id, type_dd, fault_dd, domain_dd, visiable_dd, id_select],
                    )

                    id_select.change(self.plot_signal, [id_select, dataset_state], [plot_output])

                with gr.Accordion("Run", open=False):
                    train_ids = gr.Textbox(label="Train IDs (comma separated)")
                    val_ids = gr.Textbox(label="Validation IDs")
                    test_ids = gr.Textbox(label="Test IDs")
                    agent_output = gr.Textbox(label="Agent Output")
                    run_btn = gr.Button("Run Agent")
                    run_btn.click(
                        self.process_split,
                        [train_ids, val_ids, test_ids, dataset_state, session_state],
                        [agent_output],
                    )

            with gr.Group():
                gr.Markdown("**Your request**", container=True)
                text_input = gr.Textbox(lines=3, label="Chat Message", container=False, placeholder="Enter your prompt here and press Shift+Enter or press the button")
                submit_btn = gr.Button("Submit", variant="primary")

            if self.file_upload_folder is not None:
                upload_file = gr.File(label="Upload a file")
                upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                upload_file.change(self.upload_file, [upload_file, file_uploads_log], [upload_status, file_uploads_log])

            chatbot = gr.Chatbot(
                label="Agent",
                type="messages",
                avatar_images=(
                    None,
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                ),
                resizeable=True,
                scale=1,
            )

            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input, submit_btn],
            ).then(
                self.interact_with_agent,
                [stored_messages, chatbot, session_state],
                [chatbot],
            ).then(
                lambda: (
                    gr.Textbox(interactive=True, placeholder="Enter your prompt here and press Shift+Enter or the button"),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

            submit_btn.click(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input, submit_btn],
            ).then(
                self.interact_with_agent,
                [stored_messages, chatbot, session_state],
                [chatbot],
            ).then(
                lambda: (
                    gr.Textbox(interactive=True, placeholder="Enter your prompt here and press Shift+Enter or the button"),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

        return chat_demo

    def create_kb_tab(self):
        """Return a tab for browsing knowledge base files."""
        docs = [p.name for p in self.kb_dir.glob("*") if p.suffix in {".md", ".txt"}]
        with gr.Blocks() as kb_tab:
            selector = gr.Dropdown(choices=docs, label="Document")
            content = gr.Textbox(lines=20, label="Content")
            selector.change(self.load_kb, [selector], [content])
        return kb_tab

    def create_app(self):
        """Construct and return the full Gradio application."""
        session_state = gr.State({})
        chat_tab = self.create_chat_tab(session_state)
        kb_tab = self.create_kb_tab()
        return gr.TabbedInterface(
            [chat_tab, kb_tab],
            ["Chat", "Knowledge Base"],
            title=self.name,
            theme="ocean",
        )

