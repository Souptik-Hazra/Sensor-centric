"""
plot_scm_dag.py
Publication-Grade High-Resolution Visual Generator for METR-LA Structural Causal Model (DAG).
Uses NetworkX & Matplotlib to render color-coded causal node tiers and path annotations.
"""

import os
import matplotlib.pyplot as plt
import networkx as nx
import shutil

def main():
    print("=== GENERATING PUBLICATION-GRADE STRUCTURAL CAUSAL MODEL (DAG) VISUAL ===")
    
    # 1. Initialize Directed Graph
    G = nx.DiGraph()
    
    # Define Nodes with Categories
    nodes_info = {
        'traffic_regime': {'pos': (0.0, 0.8), 'label': 'Traffic Regime (Z1)\n[Confounder]', 'color': '#38bdf8'},
        'road_type':      {'pos': (0.0, 0.2), 'label': 'Road Type (Z2)\n[Confounder]', 'color': '#38bdf8'},
        'density':        {'pos': (1.0, 0.5), 'label': 'Sensor Density (X)\n[Treatment]', 'color': '#f97316'},
        'reliability':    {'pos': (2.0, 0.8), 'label': 'Reliability (W1)\n[Mediator R_i]', 'color': '#a855f7'},
        'topology':       {'pos': (2.0, 0.2), 'label': 'Topology (W2)\n[Mediator W_ij]', 'color': '#a855f7'},
        'disparity':      {'pos': (3.0, 0.5), 'label': 'Forecast Disparity (Y)\n[Outcome Residual]', 'color': '#ef4444'}
    }
    
    for n, data in nodes_info.items():
        G.add_node(n, pos=data['pos'], label=data['label'], color=data['color'])
        
    # Define Directed Edges
    edges = [
        ('traffic_regime', 'density'),
        ('road_type', 'density'),
        ('traffic_regime', 'disparity'),
        ('road_type', 'topology'),
        ('density', 'reliability'),
        ('density', 'topology'),
        ('density', 'disparity'),
        ('reliability', 'topology'),
        ('reliability', 'disparity'),
        ('topology', 'disparity')
    ]
    G.add_edges_from(edges)
    
    # 2. Configure Matplotlib Figure
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    fig.patch.set_facecolor('#0f172a') # Sleek dark academic theme
    ax.set_facecolor('#0f172a')
    
    pos = {n: data['pos'] for n, data in nodes_info.items()}
    colors = [data['color'] for n, data in nodes_info.items()]
    labels = {n: data['label'] for n, data in nodes_info.items()}
    
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_size=3800, node_color=colors, alpha=0.92, edgecolors='#ffffff', linewidths=2, ax=ax)
    
    # Draw Labels inside Nodes
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold', font_color='#ffffff', font_family='sans-serif', ax=ax)
    
    # Custom Edge Drawing with Curved Arrows for Clutter-Free Layout
    edge_style_map = {
        ('traffic_regime', 'density'): 0.0,
        ('road_type', 'density'): 0.0,
        ('traffic_regime', 'disparity'): 0.25,
        ('road_type', 'topology'): -0.15,
        ('density', 'reliability'): 0.1,
        ('density', 'topology'): -0.1,
        ('density', 'disparity'): 0.0,
        ('reliability', 'topology'): 0.0,
        ('reliability', 'disparity'): 0.15,
        ('topology', 'disparity'): -0.15
    }
    
    for u, v in G.edges():
        rad = edge_style_map.get((u, v), 0.0)
        ax.annotate("",
                    xy=pos[v], xycoords='data',
                    xytext=pos[u], textcoords='data',
                    arrowprops=dict(arrowstyle="-|>", color="#94a3b8",
                                    connectionstyle=f"arc3,rad={rad}",
                                    mutation_scale=22, lw=2.2, alpha=0.85))
        
    # Annotate Causal Effect Percentages on Key Mediator Edges
    ax.text(2.55, 0.76, "Ctf-IE_R = 61.3%", color="#a855f7", fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#1e293b", ec="#a855f7", lw=1.5))
    ax.text(2.0, 0.50, "Ctf-DE = 21.4%", color="#f97316", fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#1e293b", ec="#f97316", lw=1.5))
    ax.text(2.55, 0.24, "Ctf-IE_T = -17.3%", color="#38bdf8", fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#1e293b", ec="#38bdf8", lw=1.5))
    
    plt.title("METR-LA Structural Causal Model (SCM) & Mediation Pathways\n[Pearl Level-3 do-Calculus & Plecko Structural Fairness Model]", 
              fontsize=13, fontweight='bold', color='#38bdf8', pad=20)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save Image
    out_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(out_dir, "scm_dag_publication_plot.png")
    plt.savefig(img_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"[OK] High-resolution SCM DAG visual generated successfully: {img_path}")
    
    # Copy to root directory if called from subdirectory
    root_copy = os.path.join(out_dir, "..", "..", "scm_dag_publication_plot.png")
    if os.path.exists(os.path.dirname(root_copy)):
        shutil.copy(img_path, root_copy)

if __name__ == '__main__':
    main()
