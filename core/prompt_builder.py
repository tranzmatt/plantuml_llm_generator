from typing import Dict, List, Any, Tuple

STRICT_RULES = """
    STRICT PLANTUML SYNTAX RULES (MANDATORY):

    1. Relationship arrows must ALWAYS follow this format:
          A --> B : "label text"
       Never place labels between A and B.
       Never write: A --> "label" B

    2. Allowed arrow forms include:
          -->     <--     --      <..     ..>     -[#color]->

       All labels go AFTER the arrow ONLY:
          A --> B : "uses"

    3. If a node name requires quotes, define an alias:
          "Image Recognition API" as ImageRecognitionAPI
          A --> ImageRecognitionAPI : "uses"

       Do NOT use quoted node names directly as endpoints.

    4. Every diagram MUST begin with @startuml and end with @enduml.

    5. Output MUST be valid for PlantUML version 1.2025.0.
    """


def build_system_prompt() -> str:
    return (
        "You are an expert PlantUML generator. You strictly follow the "
        "PlantUML 1.2025.0 syntax and the rules given.

        " + STRICT_RULES
    )


def build_user_prompt(
        repo_name: str,
        repo_text: str,
        rag_examples: Dict[str, List[Dict[str, Any]]],
) -> str:
    diagram_types = [
        "class", "sequence", "activity", "state",
        "component", "deployment", "usecase", "object",
    ]

    diagram_desc = {
        "class": "overall classes/components and their relationships in the system.",
        "sequence": "interactions over time between UI, services, and queues.",
        "activity": "workflow of main processes (e.g., image to recognition to bio/social/etc).",
        "state": "state transitions of important entities (e.g., a request lifecycle).",
        "component": "high-level components (services, queues, APIs, UI) and dependencies.",
        "deployment": "nodes/containers where services might be deployed (optional but reasonable).",
        "usecase": "user-facing use cases (e.g., user uploads image → gets celeb bio/social).",
        "object": "example runtime instances and how they relate.",
    }

    rag_section = "\n\n--- RAG EXAMPLES ---\n"
    for dtype, examples in rag_examples.items():
        rag_section += f"\n# {dtype.upper()} DIAGRAM EXAMPLES\n"
        for ex in examples:
            plantuml = ex.get("plantuml", "").strip()
            if plantuml:
                rag_section += plantuml + "\n\n"

    diagram_instructions = "\n".join(
        [f"- '{t}': {diagram_desc[t]}" for t in diagram_types]
    )

    return f"""You are analyzing the Python repository '{repo_name}'.

    Below is the concatenated source code of the repo. Use it to infer the system's architecture,
    responsibilities, external dependencies, async flows, and queues.

    Your task is to generate EIGHT valid PlantUML diagrams, one for each of the following types:

    {diagram_instructions}

    REPOSITORY SOURCE CODE (READ-ONLY CONTEXT):
    ------------------------------------------
    {repo_text}

    {rag_section}

    OUTPUT FORMAT (MANDATORY):
    --------------------------
    Return ONLY a JSON object, with EXACTLY these keys:
    "class", "sequence", "activity", "state", "component", "deployment", "usecase", "object"

    Each key's value MUST be a string containing a COMPLETE, standalone PlantUML diagram, like:

    "@startuml\n...diagram content...\n@enduml"

    Do not include any commentary, markdown, or fields other than these eight keys.
    """
