import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

# Set page configuration
st.set_page_config(page_title="Maximal Clique Analyzer", layout="wide")

# Initialize session state to keep track of algorithm progress
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.current_step = 0
    st.session_state.all_steps = []
    st.session_state.maximal_cliques = []
    st.session_state.graph = None
    st.session_state.pos = None
    st.session_state.call_stack = []
    st.session_state.page = "visualization"  # Default page

# Add navigation in sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Algorithm Visualization", "Algorithm Reports"], index=0)

# Update session state with selected page
if page == "Algorithm Visualization":
    st.session_state.page = "visualization"
else:
    st.session_state.page = "reports"

# Functions for algorithm visualization
def parse_edge_list(edge_text):
    """Parse the edge list from input text."""
    edges = []
    for line in edge_text.strip().split('\n'):
        if line.strip():
            u, v = map(int, line.strip().split())
            edges.append((u, v))
    return edges

def build_graph(edges):
    """Build a NetworkX graph from edge list."""
    G = nx.Graph()
    for u, v in edges:
        G.add_edge(u, v)
    return G

def initialize_algorithm(G):
    """Initialize the algorithm data structures."""
    steps = []
    
    # Initial state
    steps.append({
        'description': "Algorithm starts with an empty clique C = {} and all vertices as candidates.",
        'current_clique': set(),
        'candidates': set(G.nodes()),
        'action': 'initialize',
        'maximality_check': None,
        'maximality_result': None,
        'call_level': 0
    })
    
    return steps

def simulate_update(G, initial_C=None, initial_candidates=None, call_level=0):
    """Simulate the UPDATE function to generate steps."""
    if initial_C is None:
        initial_C = set()
    if initial_candidates is None:
        initial_candidates = set(G.nodes())
        
    steps = []
    
    # Add initial state for this call
    steps.append({
        'description': f"Entering UPDATE with C = {initial_C} and candidates = {initial_candidates}",
        'current_clique': initial_C.copy(),
        'candidates': initial_candidates.copy(),
        'action': 'enter_update',
        'maximality_check': None,
        'maximality_result': None,
        'call_level': call_level
    })
    
    # Empty candidates check
    if not initial_candidates:
        maximality_result = is_maximal(G, initial_C)
        steps.append({
            'description': f"Candidates empty. Testing if C = {initial_C} is maximal.",
            'current_clique': initial_C.copy(),
            'candidates': set(),
            'action': 'maximality_test',
            'maximality_check': initial_C.copy(),
            'maximality_result': maximality_result,
            'call_level': call_level
        })
        
        if maximality_result:
            steps.append({
                'description': f"C = {initial_C} is a maximal clique!",
                'current_clique': initial_C.copy(),
                'candidates': set(),
                'action': 'found_clique',
                'maximality_check': None,
                'maximality_result': None,
                'call_level': call_level
            })
        else:
            steps.append({
                'description': f"C = {initial_C} is not maximal.",
                'current_clique': initial_C.copy(),
                'candidates': set(),
                'action': 'not_maximal',
                'maximality_check': None,
                'maximality_result': None,
                'call_level': call_level
            })
            
        steps.append({
            'description': f"Returning from UPDATE with C = {initial_C}",
            'current_clique': initial_C.copy(),
            'candidates': set(),
            'action': 'return',
            'maximality_check': None,
            'maximality_result': None,
            'call_level': call_level
        })
        return steps
    
    # Process each candidate
    candidates = list(initial_candidates)
    while candidates:
        v = candidates[0]
        candidates.remove(v)
        remaining_candidates = set(candidates)
        
        steps.append({
            'description': f"Selected vertex {v} from candidates.",
            'current_clique': initial_C.copy(),
            'candidates': remaining_candidates.copy(),
            'action': 'select_vertex',
            'selected_vertex': v,
            'maximality_check': None,
            'maximality_result': None,
            'call_level': call_level
        })
        
        # Add v to C
        new_C = initial_C.copy()
        new_C.add(v)
        
        steps.append({
            'description': f"Added vertex {v} to clique C, new C = {new_C}",
            'current_clique': new_C.copy(),
            'candidates': remaining_candidates.copy(),
            'action': 'add_to_clique',
            'selected_vertex': v,
            'maximality_check': None,
            'maximality_result': None,
            'call_level': call_level
        })
        
        # Find new candidates that are adjacent to v
        new_candidates = set()
        for u in remaining_candidates:
            if G.has_edge(v, u):
                new_candidates.add(u)
        
        steps.append({
            'description': f"Filtered candidates to those adjacent to {v}, new candidates = {new_candidates}",
            'current_clique': new_C.copy(),
            'candidates': new_candidates.copy(),
            'action': 'filter_candidates',
            'selected_vertex': v,
            'maximality_check': None,
            'maximality_result': None,
            'call_level': call_level
        })
        
        # Recursive call
        recursive_steps = simulate_update(G, new_C, new_candidates, call_level + 1)
        steps.extend(recursive_steps)
        
    steps.append({
        'description': f"No more candidates to process. Returning from UPDATE with C = {initial_C}",
        'current_clique': initial_C.copy(),
        'candidates': set(),
        'action': 'return',
        'maximality_check': None,
        'maximality_result': None,
        'call_level': call_level
    })
    
    return steps

def is_maximal(G, C):
    """Check if a clique is maximal."""
    if not C:
        return False
    
    # Check if C is actually a clique
    for u in C:
        for v in C:
            if u != v and not G.has_edge(u, v):
                return False
    
    # Check if any vertex outside C can be added
    for y in G.nodes():
        if y not in C:
            can_add = True
            for x in C:
                if not G.has_edge(x, y):
                    can_add = False
                    break
            if can_add:
                return False  # Not maximal
    
    return True  # Is maximal

def find_maximal_cliques(G):
    """Find all maximal cliques in the graph."""
    maximal_cliques = []
    
    def update(C, candidates):
        if not candidates:
            if is_maximal(G, C):
                maximal_cliques.append(C.copy())
            return
        
        candidates_copy = candidates.copy()
        while candidates_copy:
            v = next(iter(candidates_copy))
            candidates_copy.remove(v)
            
            new_C = C.copy()
            new_C.add(v)
            
            new_candidates = set()
            for u in candidates_copy:
                if G.has_edge(v, u):
                    new_candidates.add(u)
            
            update(new_C, new_candidates)
    
    update(set(), set(G.nodes()))
    return maximal_cliques

# Define the Algorithm Visualization Page
def show_visualization_page():
    st.title("Maximal Clique Algorithm Visualization")
    
    # Graph Input
    st.sidebar.header("Graph Input")
    
    # Default edge list
    default_edges = """0 1
1 2
0 2
1 3
1 4
2 4
2 3
3 4
6 7
7 8
8 9
9 6
6 8
7 9"""
    
    edge_list_text = st.sidebar.text_area("Edge List (one edge per line)", value=default_edges, height=200)
    
    if st.sidebar.button("Initialize Algorithm"):
        # Parse edge list
        edges = parse_edge_list(edge_list_text)
        
        # Build graph
        G = build_graph(edges)
        
        # Generate positions for visualization
        pos = nx.spring_layout(G, seed=42)
        
        # Generate all steps
        all_steps = initialize_algorithm(G)
        all_steps.extend(simulate_update(G)[1:])  # Skip the first step which is duplicate
        
        # Find all maximal cliques
        maximal_cliques = find_maximal_cliques(G)
        
        # Save to session state
        st.session_state.initialized = True
        st.session_state.current_step = 0
        st.session_state.all_steps = all_steps
        st.session_state.maximal_cliques = maximal_cliques
        st.session_state.graph = G
        st.session_state.pos = pos
        st.session_state.total_steps = len(all_steps)
    
    # Step controls
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.session_state.initialized and st.session_state.current_step > 0:
            if st.button("Previous Step"):
                st.session_state.current_step -= 1
    
    with col2:
        if st.session_state.initialized:
            st.progress(st.session_state.current_step / max(1, len(st.session_state.all_steps) - 1))
            step_text = f"Step {st.session_state.current_step + 1}/{len(st.session_state.all_steps)}"
            st.markdown(f"<div style='text-align: center'>{step_text}</div>", unsafe_allow_html=True)
    
    with col3:
        if st.session_state.initialized and st.session_state.current_step < len(st.session_state.all_steps) - 1:
            if st.button("Next Step"):
                st.session_state.current_step += 1
    
    # Visualization area
    if st.session_state.initialized:
        col_graph, col_status = st.columns([2, 1])
        
        with col_graph:
            G = st.session_state.graph
            pos = st.session_state.pos
            step = st.session_state.all_steps[st.session_state.current_step]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Draw graph
            nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
            
            # Get node colors based on current state
            current_clique = step['current_clique']
            candidates = step['candidates']
            
            # Track position of nodes in different sets for label placement
            clique_nodes = []
            candidate_nodes = []
            other_nodes = []
            
            for node in G.nodes():
                if node in current_clique:
                    clique_nodes.append(node)
                elif node in candidates:
                    candidate_nodes.append(node)
                else:
                    other_nodes.append(node)
            
            # Draw nodes with different colors
            nx.draw_networkx_nodes(G, pos, nodelist=clique_nodes, 
                                    node_color='green', node_size=700, alpha=0.8, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=candidate_nodes, 
                                    node_color='yellow', node_size=500, alpha=0.8, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, 
                                    node_color='gray', node_size=500, alpha=0.5, ax=ax)
            
            # Highlight selected vertex if applicable
            if 'selected_vertex' in step and step['selected_vertex'] is not None:
                nx.draw_networkx_nodes(G, pos, nodelist=[step['selected_vertex']], 
                                        node_color='red', node_size=700, alpha=0.8, ax=ax)
            
            # Highlight maximality check if applicable
            if step['maximality_check'] is not None:
                check_edges = []
                check_nodes = list(step['maximality_check'])
                
                # Get all edges in the potential clique
                for i, u in enumerate(check_nodes):
                    for v in check_nodes[i+1:]:
                        if G.has_edge(u, v):
                            check_edges.append((u, v))
                
                nx.draw_networkx_edges(G, pos, edgelist=check_edges, width=3, 
                                        edge_color='green', ax=ax)
                
                # If not maximal, highlight the vertex that could be added
                if not step['maximality_result']:
                    for y in G.nodes():
                        if y not in step['maximality_check']:
                            can_add = True
                            for x in step['maximality_check']:
                                if not G.has_edge(x, y):
                                    can_add = False
                                    break
                            if can_add:
                                nx.draw_networkx_nodes(G, pos, nodelist=[y], 
                                                        node_color='purple', node_size=700, 
                                                        alpha=0.8, ax=ax)
                                # Draw edges to show why it can be added
                                connecting_edges = [(y, x) for x in step['maximality_check'] if G.has_edge(y, x)]
                                nx.draw_networkx_edges(G, pos, edgelist=connecting_edges, 
                                                        width=2.5, edge_color='purple', 
                                                        style='dashed', ax=ax)
                                break
            
            # Draw node labels
            nx.draw_networkx_labels(G, pos, ax=ax)
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=15, label='In Clique C'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=15, label='Candidates'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Selected Vertex'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=15, label='Other Vertices'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=15, label='Could Be Added (Not Maximal)')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.title("Maximal Clique Algorithm Visualization")
            plt.axis('off')
            st.pyplot(fig)
        
        with col_status:
            st.subheader("Algorithm State")
            
            # Display current clique
            st.markdown(f"**Current Clique C:** {{{', '.join(map(str, step['current_clique']))}}}")
            
            # Display candidates
            st.markdown(f"**Candidates:** {{{', '.join(map(str, step['candidates']))}}}")
            
            # Display call level
            st.markdown(f"**Recursion Depth:** {step['call_level']}")
            
            # Display step description
            st.subheader("Current Step")
            st.markdown(step['description'])
            
            # Display action type
            action_types = {
                'initialize': "Initializing algorithm",
                'enter_update': "Entering UPDATE function",
                'maximality_test': "Testing for maximality",
                'found_clique': "Found maximal clique!",
                'not_maximal': "Clique is not maximal",
                'select_vertex': "Selecting a vertex",
                'add_to_clique': "Adding vertex to clique",
                'filter_candidates': "Filtering candidates",
                'return': "Returning from function call"
            }
            
            st.markdown(f"**Action:** {action_types.get(step['action'], step['action'])}")
            
            # Display all found maximal cliques
            st.subheader("Final Results")
            st.markdown(f"**Maximal Cliques Found:** {len(st.session_state.maximal_cliques)}")
            
            clique_df = pd.DataFrame({
                'Clique': [f"{{{', '.join(map(str, clique))}}}" for clique in st.session_state.maximal_cliques]
            })
            st.dataframe(clique_df)
    
    # Display instructions if not initialized
    if not st.session_state.initialized:
        st.info("Enter the edge list in the sidebar and click 'Initialize Algorithm' to begin the visualization.")
        
        st.markdown("""
        ## How to use this visualization:
        
        1. Enter the edge list in the format "u v" (one edge per line) in the sidebar
        2. Click the "Initialize Algorithm" button
        3. Use the "Next Step" and "Previous Step" buttons to navigate through the algorithm execution
        4. The visualization will show:
            - Green nodes: Current clique C
            - Yellow nodes: Current candidates
            - Red node: Currently selected vertex
            - Purple node: Vertex that could be added (indicates clique is not maximal)
            - Green edges: Edges within the clique being tested for maximality
        """)

# Define the Report Page
def show_reports_page():
    st.title("Maximal Clique Algorithm Reports")
    
    # Create tabs for the two reports
    tab1, tab2 = st.tabs(["Tomita Algorithm", "Chiba & Nishizeki Algorithm"])
    
    with tab1:
        # Set a nice header with a colored background
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:10px; border-radius:10px; margin-bottom:20px">
            <h2 style="text-align:center; color:#1e3d59;">Tomita Algorithm for Maximal Clique Enumeration</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for better content layout
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Introduction with formatting
            st.markdown("""
            ### 📌 Introduction
            
            This paper presents a depth-first search algorithm for generating all the maximal cliques of
            an undirected graph, in which pruning methods are employed as in the Bron–Kerbosch
            algorithm. The algorithm's worst-case running time complexity is **O(3^(n/3))**, for a graph
            with n vertices. This is the best one could hope for as a function of n, since there exist up to
            3^(n/3) maximal cliques in a graph with n vertices as shown by Moon and Moser.
            """)
            
            # Algorithm description with nice formatting
            st.markdown("""
            ### 🔍 Algorithm Details
            
            The algorithm uses the following key data structures:
            - **Q**: Current clique being built. Initiated as an empty set.
            - **SUBG**: Set of vertices that can be added to Q. Initiated as a set containing all vertices V of G(V,E).
            - **CAND**: Vertices from SUBG that haven't been processed (candidates for expansion of cliques). Initiated as a set containing all vertices V of G(V,E).
            - **FINI**: Vertices already processed at a given recursion level (finished nodes). Initiated as an empty set.
            
            The core of the algorithm is a recursive procedure named **EXPAND(Q, SUBG, CAND, FINI)** that works as follows:
            
            **If SUBG is empty:**
            - The current clique being built is optimal
            - Increase num_max_cliques
            - Build clique_size_distr by increasing the count of the corresponding clique size
            
            **Else:**
            1. Choose a pivot vertex u from SUBG that maximizes ∣CAND∩Γ(u)∣, where Γ(u) denotes neighbors of vertex u
            2. Build the candidate set (CAND-Γ(u)) for expansion of Q
            3. For each vertex q in candidates set:
                - Create SUBG_q = SUBG∩Γ(q), CAND_q = CAND∩Γ(q), FINI_q as empty set
                - Add q to Q and recursively call EXPAND(Q, SUBG_q, CAND_q, FINI_q)
                - After the recursion call, remove q from Q and move it from CAND to FINI
            """)
            
            # Time complexity analysis with nice formatting
            st.markdown("""
            ### ⏱️ Time Complexity Analysis
            
            The time complexity can be broken down as follows:
            1. Choosing a pivot from SUBG - **O(n²)**
            2. Generating Candidate set - **O(n)**
            3. Process each vertex in candidate set (|CAND-Γ(u)|)
               - For each vertex, a recursive call is made on a subproblem with fewer vertices
               - Worst case scenario: subproblem is of size n-1, giving **O(n-1)**
            
            This leads to the recurrence relation:
            
            **T(n) = k⋅T(n-1) + O(n²) + O(n)**
            
            Where k = |CAND-Γ(u)| represents the branching factor.
            
            The Moon-Moser theorem states that the maximum number of maximal cliques in an
            undirected graph with n vertices is 3^(n/3), when every vertex belongs to exactly three
            maximal cliques. Therefore, in the worst case, the branching factor is 3. The new recurrence
            relation becomes:
            
            **T(n) = 3⋅T(n-1) + O(n²)**
            
            Solving this gives us **T(n) = O(3^(n/3))** as the overall time complexity.
            """)
        
        with col2:
            # Experimental Results in a card-like container
            st.markdown("""
            <div style="background-color:#ffffff; padding:15px; border-radius:10px; border:1px solid #e0e0e0; margin-bottom:20px">
                <h3 style="text-align:center; color:#1e3d59;">📊 Experimental Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a nicer looking table for the results
            results_data = {
                "Dataset": ["Wiki-Vote", "Email-Enron", "As-Skitter"],
                "Total Maximal Cliques": ["459,002", "226,859", "NaN"],
                "Execution Time (seconds)": ["122", "624", "NaN"],
                "Largest Clique Size": ["17", "20", "NaN"]
            }
            
            df = pd.DataFrame(results_data)
            
            # Apply custom styling to the table
            st.dataframe(df, use_container_width=True)
            
            # References
            st.markdown("""
            <div style="background-color:#ffffff; padding:15px; border-radius:10px; border:1px solid #e0e0e0; margin-top:20px">
                <h3 style="text-align:center; color:#1e3d59;">📚 References</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            1. Tomita, Etsuji & Tanaka, Akira & Takahashi, Haruhisa. (2006). The Worst-Case Time Complexity for Generating All Maximal Cliques and Computational Experiments. Theoretical Computer Science. 363. 28-42. 10.1016/j.tcs.2006.06.015.
            
            2. C. Bron, J. Kerbosch, Algorithm 457, finding all cliques of an undirected graph, Comm. ACM 16 (1973) 575–577.
            
            3. J.W. Moon, L. Moser, On cliques in graphs, Israel J. Math. 3 (1965) 23–28.
            """)
            
            # Add a visualization of time complexity growth
            st.markdown("""
            <div style="background-color:#ffffff; padding:15px; border-radius:10px; border:1px solid #e0e0e0; margin-top:20px">
                <h3 style="text-align:center; color:#1e3d59;">📈 Time Complexity Visualization</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a simple visualization of the O(3^n) growth
            fig, ax = plt.subplots(figsize=(8, 4))
            n_values = np.arange(1, 11)
            complexity = 3 ** (n_values / 3)
            ax.plot(n_values, complexity, marker='o', color='#1e3d59', linewidth=2)
            ax.set_xlabel('Number of Vertices (n)')
            ax.set_ylabel('Time Complexity O(3^(n/3))')
            ax.set_title('Growth of Time Complexity')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_yscale('log')
            st.pyplot(fig)
    
    with tab2:
        # Set a nice header with a colored background
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:10px; border-radius:10px; margin-bottom:20px">
            <h2 style="text-align:center; color:#1e3d59;">Chiba and Nishizeki Algorithm Analysis</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Introduction with better formatting
        st.markdown("""
        ### 📌 Introduction
        
        The algorithm for listing all maximal cliques in a graph, as described in the paper, was implemented and
        tested on three real-world datasets: the Wiki-Vote dataset, Email-Enron dataset, and Autonomous Systems
        by Skitter datasets. This report presents an analysis of the algorithm's performance, including the total
        number of maximal cliques found, execution time, and additional metrics for evaluating computational
        efficiency.
        """)
        
        # Create two columns for better content layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ### 🔍 Key Parts of the Algorithm
            
            The Chiba-Nishizeki algorithm lists all maximal cliques in a graph using a vertex
            elimination approach combined with adjacency list traversal. The key steps are:
            
            1. **Vertex Sorting**: We sort the adjacency list such that the nodes with least degree appear first and
               thus are ordered based on degree.
            
            2. **Clique Expansion**: For each node, the algorithm finds cliques that include the vertex by searching
               its neighborhood.
            
            3. **Maximality Test**: For a given clique C, we try to add a node and check if it can be made maximal
               by adding that node. Doing this test for all nodes ensures that the current clique is maximal and
               no more edges can be added to it.
            
            4. **Backtracking & Pruning**: Unnecessary computations are avoided by eliminating processed
               vertices, ensuring (using the lexicographical test) each clique is enumerated exactly once.
            """)
            
            st.markdown("""
            ### ⏱️ Time Complexity Breakdown
            
            The algorithm operates with a worst-case time complexity of **O(α(G) * m)** per maximal clique, where:
            - **m** is the number of edges in the graph.
            - **α(G)** is the arboricity of the graph
            
            #### Step-wise Complexity Analysis
            
            1. **readGraph (O(m))**: Simply reading all edges from file and updating the adjacency lists.
            
            2. **Vertex Sorting (O(n))**:
               - Uses bucket sort algorithm as degrees and node numbers are discrete integer values.
            
            3. **Building Neighborhood Subgraphs (O(m))**:
               - Before a node is added, it goes through maximalityTest() and lexicographicTest(), each
                 of which take O(m) time.
            
            4. **Recursive Clique Enumeration (O(α(G) * m))**:
               - The recursion ensures that each clique is found exactly once, using edge-based expansion
                 limited by arboricity α(G). This is implemented in the UPDATE() function.
            
            5. **Pruning & Backtracking (O(m))**:
               - By removing already processed vertices, the search space is reduced.
            
            Combining these steps, the overall complexity per maximal clique remains **O(α(G) * m)**.
            """)
        
        with col2:
            # Experimental Results in a card-like container
            st.markdown("""
            <div style="background-color:#ffffff; padding:15px; border-radius:10px; border:1px solid #e0e0e0; margin-bottom:20px">
                <h3 style="text-align:center; color:#1e3d59;">📊 Experimental Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a nicer looking table for the results
            chiba_results = {
                "Dataset": ["Wiki-Vote", "Email-Enron", "As-Skitter"],
                "Total Maximal Cliques": ["459,002", "226,859", "37,322,355"],
                "Execution Time (seconds)": ["484", "697", "~15 days"],
                "Largest Clique Size": ["17", "20", "67"]
            }
            
            chiba_df = pd.DataFrame(chiba_results)
            
            # Apply styling
            st.dataframe(chiba_df, use_container_width=True)
            
            # Possible Optimizations section
            st.markdown("""
            <div style="background-color:#ffffff; padding:15px; border-radius:10px; border:1px solid #e0e0e0; margin-top:20px">
                <h3 style="text-align:center; color:#1e3d59;">🔧 Possible Optimizations</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            1. **2-Clique Detection Optimization**:
               - Many 2-sized cliques in the dataset are simply edges with 2 nodes of degree 1
               - These can be detected during bucket sort and removed safely
               - With β as the fraction of 2-Cliques, new time complexity is O(β·α(G)·m) per maximal clique
               - In the Skitter dataset, β = 2,319,807/37,322,355 = 0.062 (a small but worthwhile improvement)
            
            2. **Reduce Copy Operations**:
               - Optimization opportunity exists to eliminate copy operations during recursive calls
               - Using iterators instead could save significant processing time
            """)
            
            # Algorithm performance comparison visualization
            st.markdown("""
            <div style="background-color:#ffffff; padding:15px; border-radius:10px; border:1px solid #e0e0e0; margin-top:20px">
                <h3 style="text-align:center; color:#1e3d59;">📈 Algorithm Comparison</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a comparison of the two algorithms
            fig, ax = plt.subplots(figsize=(8, 4))
            
            datasets = ["Wiki-Vote", "Email-Enron"]
            tomita_times = [122, 624]
            chiba_times = [484, 697]
            
            x = np.arange(len(datasets))
            width = 0.35
            
            ax.bar(x - width/2, tomita_times, width, label='Tomita', color='#1e3d59')
            ax.bar(x + width/2, chiba_times, width, label='Chiba', color='#f5b461')
            
            ax.set_ylabel('Execution Time (seconds)')
            ax.set_title('Algorithm Execution Time Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(datasets)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')
            
            st.pyplot(fig)

# Update the main flow to choose between pages
if __name__ == "__main__":
    if st.session_state.page == "visualization":
        show_visualization_page()
    else:
        show_reports_page()
