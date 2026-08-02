import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
# Set up the page configuration
st.set_page_config(page_title="Campus Bus Route Optimizer", layout="wide")
st.title("🚌 Campus Bus Route Optimizer")
st.caption("A Graph-Based Mini Project using Python, Streamlit & NetworkX")
# ---- File Paths (works no matter where you run the command from) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "sample_data", "routes.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")

# ---- Load Data ----
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# ---- Build Graph ----
def build_graph(data):
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_edge(
            row["Source"],
            row["Destination"],
            distance=row["Distance_km"],
            density=row["Student_Density"],
            time=row["Travel_Time_min"]
        )
    return G

G = build_graph(df)

# ---- Add "AI cost" to each edge (distance + density combined) ----
def add_cost_attribute(graph, alpha=0.5, beta=0.5):
    for u, v, data in graph.edges(data=True):
        data["cost"] = alpha * data["distance"] + beta * (data["density"] / 100)
    return graph

G = add_cost_attribute(G)

# ---- Route Calculation Functions ----
def get_shortest_path(graph, source, destination):
    return nx.shortest_path(graph, source=source, target=destination, weight="distance")

def get_ai_recommended_path(graph, source, destination):
    return nx.shortest_path(graph, source=source, target=destination, weight="cost")

def get_path_stats(graph, path):
    total_distance = 0
    total_time = 0
    for i in range(len(path) - 1):
        edge_data = graph[path[i]][path[i + 1]]
        total_distance += edge_data["distance"]
        total_time += edge_data["time"]
    return total_distance, total_time
# Create 5 tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏠 Home", "🤖 AI/LLM Layer", "📘 Subjects", "🚌 Program", "📊 Outcome"]
)

# Tab 1: Home
with tab1:
    st.header("🏠 Home")
    st.subheader("Campus Bus Route Optimizer")
    
    st.write("""
    Welcome to the **Campus Bus Route Optimizer** — a mini-project that uses 
    graph theory and basic AI logic to find the most efficient bus routes 
    between campus stops.
    """)
    
    st.markdown("### 🎯 Project Objective")
    st.write("""
    Many college campuses have multiple bus stops connecting hostels, academic 
    blocks, libraries, and canteens. Students often don't know the fastest or 
    shortest route between two points. This project aims to:
    - Model all bus stops and routes as a **graph**
    - Calculate the **shortest path** between any two stops
    - Recommend the **best route** using student density and distance as factors
    """)
    
    st.markdown("### 🌍 Real-World Applications")
    st.write("""
    - Public transport apps (like Google Maps) use similar shortest-path algorithms
    - Ride-sharing apps (Uber/Ola) use graph optimization to reduce travel time
    - Logistics companies use this to plan delivery routes efficiently
    - Campus administrations can use this to reduce bus congestion and travel time
    """)

# Tab 2: AI/LLM Layer
with tab2:
    st.header("🤖 AI/LLM Layer")
    
    st.write("""
    This layer explains how simple AI logic is used to recommend the best 
    bus route — not just the shortest one, but the most *efficient* one 
    considering student crowding too.
    """)
    
    st.markdown("### 🧠 How AI Helps Here")
    st.write("""
    Instead of only picking the route with the least distance, we combine 
    **distance** and **student density** into a single score. A route that 
    is short but overcrowded may actually take longer in real life due to 
    boarding delays — so our formula accounts for that.
    """)
    
    st.markdown("### 📐 The Cost Formula")
    st.latex(r"Cost = \alpha \times Distance + \beta \times StudentDensity")
    
    st.write("""
    - **α (alpha)** and **β (beta)** are weight values between 0 and 1 that 
      control how much importance we give to distance vs. crowding
    - Lower **Cost** = better recommended route
    """)
    
    st.markdown("### 🗺️ Shortest Path Concept")
    st.latex(r"Path_{optimal} = \min_{P} \sum_{(u,v) \in P} weight(u,v)")
    
    st.write("""
    This means: out of all possible paths between a source and destination, 
    we select the one where the total distance is the smallest. This is 
    calculated using **graph-based optimization** with the NetworkX library, 
    which we'll implement in the Program tab.
    """)
# Tab 3: Subjects
with tab3:
    st.header("📘 Subjects Integration")
    
    st.write("""
    This project connects concepts from three first-year subjects. 
    Here's how each one plays a role:
    """)
    
    st.markdown("### 1️⃣ Differential Equations")
    st.write("""
    Student density at a bus stop changes over time (more students gather 
    closer to class-end times). This "rate of change" is conceptually 
    described using differential equations:
    """)
    st.latex(r"\frac{dN}{dt} = \text{rate of change of students } N \text{ over time } t")
    st.write("""
    While this project uses static density values for simplicity, this 
    concept forms the theoretical basis for future improvements like 
    real-time crowd prediction.
    """)
    
    st.markdown("### 2️⃣ Linear Algebra")
    st.write("""
    - **Vector**: A route is represented as a vector of features, e.g. 
      `[distance, student_density]`
    - **Matrix**: All bus stops and their distances can be represented as 
      an **Adjacency Matrix** — a grid where each cell (i, j) shows the 
      distance between stop i and stop j
    - **Array**: NumPy stores all of this as arrays, enabling fast 
      mathematical operations across the entire route network
    """)
    
    st.markdown("### 3️⃣ Programming in Python")
    st.write("""
    - **Functions** organize our route-finding logic into reusable blocks
    - **Loops & conditionals** let us compare multiple routes and pick the best
    - **Libraries** (Pandas, NumPy, NetworkX) let us build a working 
      optimizer without writing graph algorithms from scratch
    """)
    
    st.markdown("### 🔗 Concept Summary Table")
    st.table({
        "Subject": ["Differential Equations", "Linear Algebra", "Python Programming"],
        "Concept Used": ["Rate of change (dN/dt)", "Vectors, Matrices, Arrays", "Functions, Loops, Libraries"],
        "Application": ["Crowd density trends", "Route/distance representation", "Building the optimizer logic"]
    })
# Tab 4: Program
with tab4:
    st.header("🚌 Program")
    st.write("This tab reads real route data, builds a graph, and calculates optimized paths.")

    st.subheader("📄 Raw Route Data (from CSV)")
    st.dataframe(df)

    st.subheader("🔀 Graph Structure")
    st.write(f"Total Bus Stops (Nodes): **{G.number_of_nodes()}**")
    st.write(f"Total Direct Routes (Edges): **{G.number_of_edges()}**")

    st.subheader("🧮 Try a Route Calculation")
    stops = sorted(G.nodes())
    col1, col2 = st.columns(2)
    with col1:
        source = st.selectbox("Select Source Stop", stops, key="program_source")
    with col2:
        destination = st.selectbox("Select Destination Stop", stops, index=1, key="program_dest")

    if source == destination:
        st.warning("Please select two different stops.")
    else:
        shortest = get_shortest_path(G, source, destination)
        ai_path = get_ai_recommended_path(G, source, destination)

        st.write("**Shortest Path (by distance only):**")
        st.success(" → ".join(shortest))

        st.write("**AI-Recommended Path (distance + student density):**")
        st.success(" → ".join(ai_path))

        if shortest == ai_path:
            st.info("Both methods agree — this route is efficient AND not crowded.")
        else:
            st.info("The AI recommends a different route to avoid crowding, even if it's not the absolute shortest.")
# Tab 5: Outcome
with tab5:
    st.header("📊 Outcome")
    st.write("Select a source and destination to see the optimized route visualized on the campus bus network.")

    stops = sorted(G.nodes())
    col1, col2 = st.columns(2)
    with col1:
        source = st.selectbox("Source", stops, key="outcome_source")
    with col2:
        destination = st.selectbox("Destination", stops, index=1, key="outcome_dest")

    if source == destination:
        st.warning("Please select two different stops.")
    else:
        path = get_ai_recommended_path(G, source, destination)
        total_distance, total_time = get_path_stats(G, path)

        m1, m2 = st.columns(2)
        m1.metric("Total Distance", f"{total_distance:.2f} km")
        m2.metric("Estimated Travel Time", f"{total_time:.0f} min")

        st.write("**Optimized Route:** " + " → ".join(path))

        st.subheader("🗺️ Route Visualization")
        fig, ax = plt.subplots(figsize=(8, 5))
        pos = nx.spring_layout(G, seed=42)

        nx.draw(G, pos, with_labels=True, node_color="lightgray",
                edge_color="lightgray", node_size=1500, font_size=8, ax=ax)

        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="orange", node_size=1500, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=3, ax=ax)

        st.pyplot(fig)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(OUTPUT_DIR, "route_result.png"))
        st.caption("This route image is also saved in the output/ folder.")