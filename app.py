"""Enhanced Gradio application for the PHM agent."""

from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd
import matplotlib.pyplot as plt

from data.loader import PHMDataset
import gradio as gr

from smolagents import GradioUI
from smolagents.models import (
    InferenceClientModel,
    LiteLLMModel,
    OpenAIServerModel,
)

from agents_config import create_manager_agent


class EnhancedGradioUI(GradioUI):
    """Custom UI with dataset viewer and knowledge base tabs."""

    def __init__(self, agent, kb_dir: str = "knowledge_base", **kwargs) -> None:
        super().__init__(agent, **kwargs)
        self.kb_dir = Path(kb_dir)
        self.metadata_path: str | None = None
        self.data_path: str | None = None

    # ------------------------------------------------------------------
    # Model configuration helpers
    # ------------------------------------------------------------------
    def _build_model(self, provider: str, model_id: str, api_key: str):
        if provider == "LiteLLM":
            return LiteLLMModel(model_id=model_id, api_key=api_key)
        if provider == "OpenAI":
            return OpenAIServerModel(model_id=model_id, api_key=api_key)
        if provider == "HuggingFace":
            return InferenceClientModel(model_id=model_id, token=api_key)
        raise gr.Error(f"Unsupported provider: {provider}")

    def init_model(self, provider, model_id, api_key, session_state):
        if not api_key:
            raise gr.Error("API key required")
        model = self._build_model(provider, model_id, api_key)
        session_state["agent"] = create_manager_agent(model)
        return "Model initialized"

    # ------------------------------------------------------------------
    # Dataset handlers
    # ------------------------------------------------------------------
    def init_dataset(self, metadata_file, data_file):
        """Initialize :class:`PHMDataset` from uploaded files."""
        if metadata_file is None or data_file is None:
            raise gr.Error("Upload metadata and data files")
        ds = PHMDataset(metadata_file.name, data_file.name)
        self.metadata_path = metadata_file.name
        self.data_path = data_file.name
        ids = ds.metadata["id"].tolist()
        return ds.metadata, ds, gr.Dropdown.update(
            choices=[str(i) for i in ids], value=str(ids[0]) if ids else None
        )

    def plot_signal(self, selected_id, dataset):
        """Plot a signal from ``dataset`` by ``selected_id``."""
        if dataset is None or selected_id is None:
            raise gr.Error("Dataset not ready")
        data = dataset.load(int(selected_id))
        fig, ax = plt.subplots()
        ax.plot(data)
        ax.set_title(f"Signal {selected_id}")
        return fig

    def process_split(self, train_ids, val_ids, test_ids, dataset, session_state):
        """Send the selected IDs to the PHM agent for processing."""
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
        with gr.Blocks() as chat_demo:
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])

            with gr.Sidebar():
                gr.Markdown(
                    f"# {self.name.replace('_', ' ').capitalize()}"
                    "\n> Chat with the PHM agent"
                )

                with gr.Accordion("Model Configuration", open=False):
                    provider = gr.Dropdown(
                        ["LiteLLM", "OpenAI", "HuggingFace"],
                        label="Provider",
                        value="LiteLLM",
                    )
                    model_id = gr.Textbox("gemini/gemini-pro", label="Model ID")
                    api_key = gr.Textbox(label="API Key", type="password")
                    init_btn = gr.Button("Initialize Model")
                    status = gr.Textbox(label="Status", interactive=False)
                    init_btn.click(
                        self.init_model,
                        [provider, model_id, api_key, session_state],
                        [status],
                    )
                with gr.Accordion("Dataset", open=False):
                    metadata_file = gr.File(label="Metadata (.xls/.xlsx)")
                    data_file = gr.File(label="Data (.h5)")
                    metadata_table = gr.DataFrame(label="Metadata")
                    id_select = gr.Dropdown(label="Select ID")
                    plot_output = gr.Plot()
                    run_btn = gr.Button("Run Agent")
                    train_ids = gr.Textbox(label="Train IDs (comma separated)")
                    val_ids = gr.Textbox(label="Validation IDs")
                    test_ids = gr.Textbox(label="Test IDs")
                    agent_output = gr.Textbox(label="Agent Output")

                    metadata_file.change(
                        self.init_dataset,
                        [metadata_file, data_file],
                        [metadata_table, session_state, id_select],
                    )
                    data_file.change(
                        self.init_dataset,
                        [metadata_file, data_file],
                        [metadata_table, session_state, id_select],
                    )
                    id_select.change(
                        self.plot_signal, [id_select, session_state], [plot_output]
                    )
                    run_btn.click(
                        self.process_split,
                        [train_ids, val_ids, test_ids, session_state],
                        [agent_output],
                    )

                with gr.Group():
                    gr.Markdown("**Your request**", container=True)
                    text_input = gr.Textbox(
                        lines=3,
                        label="Chat Message",
                        container=False,
                        placeholder="Enter your prompt here and press Shift+Enter or press the button",
                    )
                    submit_btn = gr.Button("Submit", variant="primary")

                if self.file_upload_folder is not None:
                    upload_file = gr.File(label="Upload a file")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                    upload_file.change(
                        self.upload_file,
                        [upload_file, file_uploads_log],
                        [upload_status, file_uploads_log],
                    )

                # gr.HTML(
                #     "<br><br><h4><center>Powered by <a target='_blank' href='https://github.com/huggingface/smolagents'><b>smolagents</b></a></center></h4>"
                # )

            chatbot = gr.Chatbot(
                label="Agent",
                type="messages",
                avatar_images=(
                    None,
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                ),
                resizeable=True,
                scale=1,
                latex_delimiters=[
                    {"left": r"$$", "right": r"$$", "display": True},
                    {"left": r"$", "right": r"$", "display": False},
                    {"left": r"\[", "right": r"\]", "display": True},
                    {"left": r"\(", "right": r"\)", "display": False},
                ],
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
                    gr.Textbox(
                        interactive=True,
                        placeholder="Enter your prompt here and press Shift+Enter or the button",
                    ),
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
                    gr.Textbox(
                        interactive=True,
                        placeholder="Enter your prompt here and press Shift+Enter or the button",
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

        return chat_demo

    def create_dataset_tab(self, session_state):
        with gr.Blocks() as dataset_tab:
            dataset_state = gr.State(None)

            with gr.Row():
                metadata_file = gr.File(label="Metadata (.xls/.xlsx)")
                data_file = gr.File(label="Data (.h5)")

            metadata_table = gr.DataFrame(label="Metadata")
            id_select = gr.Dropdown(label="Select ID")
            plot_output = gr.Plot()

            train_ids = gr.Textbox(label="Train IDs (comma separated)")
            val_ids = gr.Textbox(label="Validation IDs")
            test_ids = gr.Textbox(label="Test IDs")
            agent_output = gr.Textbox(label="Agent Output")
            run_btn = gr.Button("Run Agent")

            metadata_file.change(
                self.init_dataset,
                [metadata_file, data_file],
                [metadata_table, dataset_state, id_select],
            )
            data_file.change(
                self.init_dataset,
                [metadata_file, data_file],
                [metadata_table, dataset_state, id_select],
            )

            id_select.change(self.plot_signal, [id_select, dataset_state], [plot_output])
            run_btn.click(
                self.process_split,
                [train_ids, val_ids, test_ids, dataset_state, session_state],
                [agent_output],
            )

        return dataset_tab

    def create_kb_tab(self):
        docs = [p.name for p in self.kb_dir.glob("*") if p.suffix in {".md", ".txt"}]
        with gr.Blocks() as kb_tab:
            selector = gr.Dropdown(choices=docs, label="Document")
            content = gr.Textbox(lines=20, label="Content")
            selector.change(self.load_kb, [selector], [content])
        return kb_tab

    def create_app(self):
        session_state = gr.State({})
        chat_tab = self.create_chat_tab(session_state)
        dataset_tab = self.create_dataset_tab(session_state)
        kb_tab = self.create_kb_tab()
        return gr.TabbedInterface(
            [chat_tab, dataset_tab, kb_tab],
            ["Chat", "Dataset", "Knowledge Base"],
            title=self.name,
            theme="ocean",
        )


def main() -> None:
    model = LiteLLMModel(model_id="gemini/gemini-2.5-pro")
    agent = create_manager_agent(model)
    ui = EnhancedGradioUI(agent, file_upload_folder="./uploads")
    ui.launch()


if __name__ == "__main__":
    main()

