### **Warming up: Setting Up the Laboratory**

Before we embark on our theoretical journey, we must first assemble our laboratory. In computational science, the laboratory is not a room of beakers and burners, but a well-defined, reproducible, and robust development environment. The principles of rigor and precision that we will apply to our algorithms must first be applied to our tools. A haphazard setup leads to inscrutable errors and frustrates the process of inquiry. A clean, version-controlled environment, by contrast, allows us to focus on the intellectual heart of our work.

This chapter is a practical guide to configuring your local machine to run the Deep Research Agent. We will not merely provide a list of commands; we will explain the purpose of each component, ensuring you understand not just *how* to build your laboratory, but *why* it is constructed in this manner.

#### **A.1 Prerequisites: The Foundational Tools**

Before we begin, ensure you have the following foundational software installed on your system. These are the bedrock upon which our more specialized environment will be built.

*   **Python:** The agent is written in Python. You will need a modern version (3.10 or newer).
*   **Git:** Our project is managed with the Git version control system. You will need it to acquire the source code.
*   **Docker:** The agent relies on an external, locally-hosted search appliance. We will use Docker to manage this service in a clean, isolated container.

#### **A.2 Acquiring the Source Code**

The first step is to obtain a local copy of the project's source code. Open a terminal and execute the following Git command:

```bash
git clone https://github.com/VladPrytula/DeepResearchHybrid
cd DeepResearchHybrid
```

This command performs two actions:
1.  `git clone`: It contacts the remote repository, downloads the entire project history and the latest versions of all files, and places them in a new directory on your local machine.
2.  `cd`: It navigates your terminal's current working directory into this new project folder. All subsequent commands should be run from this location.

#### **A.3 Dependency Management with `uv`**

A Python project rarely exists in isolation; it depends on a constellation of third-party libraries (e.g., `numpy`, `scikit-learn`, `rich`). Managing these dependencies is paramount for reproducibility. A common source of error is running code with a different version of a library than the one it was developed with.

To prevent this, we will use a **virtual environment**. This creates an isolated Python environment for our project, ensuring that the libraries we install for the agent do not conflict with any other Python projects on your system.

While many tools exist for this, we have chosen **`uv`**, a modern, extremely fast Python package installer and resolver developed in Rust. Its speed and strictness align with our philosophy of rigor and efficiency.

1.  **Install `uv`:** If you do not have `uv` installed, follow the official installation instructions from its documentation. A common method is:
    ```bash
    # On macOS / Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Create and Activate the Virtual Environment:** From the root of the project directory, run:
    ```bash
    # Create a virtual environment named .venv in the current directory
    uv venv

    # Activate the environment (the command differs for your shell)
    # For bash/zsh on macOS/Linux:
    source .venv/bin/activate
    # For Powershell on Windows:
    # .venv\Scripts\Activate.ps1
    ```
    Upon activation, you will typically see the name of the environment (`.venv`) appear in your terminal prompt, indicating that any subsequent `python` or `pip` commands will operate within this isolated space.

3.  **Synchronize Dependencies:** Now, install the exact versions of all required libraries as specified in the `requirements.txt` file.
    ```bash
    uv pip sync requirements.txt
    ```
    **Pedagogical Remark:** We use `uv pip sync` instead of the more common `pip install -r requirements.txt`. This is a deliberate choice. `sync` is stricter: it ensures that the environment contains *only* the packages listed in `requirements.txt`, removing any that might have been installed previously. This guarantees that your environment is a perfect, clean replica of the one defined by the project, enhancing reproducibility.

Your Python environment is now prepared.

#### **A.4 Provisioning the Local Search Appliance: SearXNG**

Our agent requires access to the web to gather information. Instead of relying on the proprietary, rate-limited, and often costly APIs of commercial search engines, we will provision our own local **metasearch engine**. A metasearch engine does not crawl the web itself; it acts as a privacy-preserving proxy, forwarding a query to numerous other search engines and aggregating the results.

We will use **SearXNG**, a popular, open-source, and highly configurable metasearch engine. Running it locally in a Docker container provides us with a private, fast, and reliable search service that our agent can query without limitation.

1.  **The Docker Command:** Execute the following command in your terminal. This will download the official SearXNG image (if you don't have it already) and start it as a background service.
    ```bash
    docker run --rm \
             -d -p 9090:8080 \
             -v "${PWD}/my-local-serach/searxng:/etc/searxng" \
             -e "BASE_URL=http://localhost:9090/" \
             -e "INSTANCE_NAME=deep-research-agent" \
             searxng/searxng
    ```

2.  **Dissecting the Command (The "No Steps Skipped" Mandate):** Let us analyze each part of this command to understand precisely what it does.
    *   `docker run`: The fundamental command to create and start a new Docker container.
    *   `--rm`: This is a convenience flag. It tells Docker to automatically remove the container when it is stopped, preventing a build-up of unused containers on your system.
    *   `-d`: Stands for "detached." It runs the container in the background, freeing up your terminal for other commands.
    *   `-p 9090:8080`: This is the port mapping. It maps port `9090` on your host machine (your computer) to port `8080` inside the container. SearXNG listens on port `8080` by default within its container. This mapping means you can access the service by navigating to `http://localhost:9090` in your web browser.
    *   `-v "${PWD}/my-local-serach/searxng:/etc/searxng"`: This is a volume mount. It links the `my-local-serach/searxng` directory from our project folder on your host machine to the `/etc/searxng` directory inside the container. This is crucial because it allows us to provide our own `settings.yml` file to the container, overriding its default configuration. The agent's performance is sensitive to these settings.
    *   `-e "BASE_URL=..."`: This sets an environment variable inside the container. SearXNG uses `BASE_URL` to correctly construct URLs for itself. We must set it to the public-facing URL, which is `http://localhost:9090/`.
    *   `-e "INSTANCE_NAME=..."`: This sets a descriptive name for the instance.
    *   `searxng/searxng`: This is the name of the official Docker image to use.

3.  **Verification:** After running the command, open a web browser and navigate to `http://localhost:9090`. You should see the SearXNG search interface. Your local search appliance is now operational.

#### **A.5 Configuring the Agent's Access Credentials**

The final setup step is to provide the agent with the necessary credentials to access external AI services—specifically, your Azure OpenAI endpoints for chat/generation models and for embedding models.

It is a critical security and software engineering principle to **never hard-code secrets** (like API keys) directly into the source code. Instead, we use a `.env` file, which is read by the application at runtime but is excluded from version control (via the `.gitignore` file).

1.  **Create the `.env` file:** In the root directory of the project, create a new file named `.env`.

2.  **Populate the file:** Copy the following template into your new `.env` file and replace the placeholder values (`xxxx`) with your actual credentials from the Azure portal.

    ```dotenv
    # .env

    # --- CORE AZURE API SETTINGS ---
    # This is the API version for most Azure OpenAI calls.
    AZURE_OPENAI_API_VERSION="2023-07-01-preview"

    # --- CHAT MODEL CONFIGURATION (e.g., GPT-4o) ---
    # Find these values in your Azure OpenAI resource for chat models.
    AZURE_CHAT_ENDPOINT="https://your-chat-resource.openai.azure.com/"
    AZURE_CHAT_API_KEY="xxxx_your_chat_api_key_xxxx"
    AZURE_OPENAI_DEPLOYMENT="your-gpt4-deployment-name"
    AGENT_SUMMARY_MODEL="your-gpt4-deployment-name" # Often the same as above

    # --- EMBEDDING MODEL CONFIGURATION (e.g., text-embedding-3-small) ---
    # Find these values in your Azure OpenAI resource for embedding models.
    AZURE_EMBEDDING_ENDPOINT="https://your-embedding-resource.openai.azure.com/"
    AZURE_EMBEDDING_API_KEY="xxxx_your_embedding_api_key_xxxx"
    AZURE_EMBEDDING_DEPLOYMENT="your-embedding-deployment-name"

    # --- LOCAL SEARCH ENGINE ---
    # This should match the port you exposed in the Docker command.
    SEARX_URL="http://127.0.0.1:9090/search?q="

    # --- LOGGING LEVEL ---
    # Can be DEBUG, INFO, WARNING, ERROR
    LOG_LEVEL="INFO"
    ```

*   **Explanation of Key Variables:**
    *   `AZURE_CHAT_ENDPOINT` / `AZURE_CHAT_API_KEY` / `AZURE_OPENAI_DEPLOYMENT`: These direct the agent to your deployed generation model (like GPT-4o). The `ENDPOINT` is the URL of your Azure OpenAI resource, the `API_KEY` is your secret access key, and the `DEPLOYMENT` is the specific name you gave your model when you deployed it in Azure AI Studio.
    *   `AZURE_EMBEDDING_ENDPOINT` / `AZURE_EMBEDDING_API_KEY` / `AZURE_EMBEDDING_DEPLOYMENT`: These point to your embedding model (like `text-embedding-3-small` or `text-embedding-ada-002`). It is common practice in Azure to have separate resources and deployments for chat and embedding models.
    *   `SEARX_URL`: This tells the agent where to find the local search appliance we just started with Docker. The port (`9090`) must match the one used in the `docker run -p` flag.

#### **A.6 Running the Agent**

Your laboratory is now fully operational. The source code is present, dependencies are isolated and installed, the search appliance is running, and the agent's credentials are in place.

To initiate a research task, execute the `main.py` script from your activated virtual environment:

```bash
python main.py "What are the core arguments in favor of and against Strong AI?"
```

You can control the verbosity of the output using the `--output-style` flag:

*   `--output-style summary`: The default, providing high-level updates after each cycle.
*   `--output-style detailed`: Shows all verbose logs, useful for debugging.
*   `--output-style progress`: Shows a minimal progress bar, ideal for long runs.

#### **A.7 A Note on the Secondary Objective: The MCP Server**

The file `mcp_server.py` serves the project's second goal: wrapping our entire `ResearchPipeline` into a tool that can be called by other systems, such as the OpenWebUI. The `mcp.tool()` decorator exposes the `deep_research` function, which in turn calls our `run_deep_research` API. Running `python mcp_server.py` will start this server, allowing compatible LLM orchestrators to discover and use our agent as a powerful research tool. We will explore this system-level integration in our final chapter.
> `uvx mcpo --host 127.0.0.1 --port 8000 -- uv run python mcp_server.py`

With our environment now meticulously configured, we are prepared to delve back into the theoretical and implementational details of the agent's mind. Let us proceed.