"""
plot_system_architecture.py
Generates a publication-grade 300 DPI visual System Architecture Diagram for the
METR-LA Structural Causal Digital Twin Framework.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil

def main():
    print("=== GENERATING PUBLICATION-GRADE SYSTEM ARCHITECTURE DIAGRAM ===")
    
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    fig.patch.set_facecolor('#0f172a') # Dark academic theme
    ax.set_facecolor('#0f172a')
    
    # Layer Box Specifications
    layers = [
        {
            'title': 'LAYER 1: WEBGL GIS OPERATIONS CONSOLE (JavaScript / HTML5 / Leaflet)',
            'desc': 'Interactive 60 FPS GIS Map  |  3-Horizon Selector (15m | 30m | 60m)  |  288-Step Replay Slider  |  Fault Stimulus Injector',
            'xy': (0.05, 0.77), 'width': 0.90, 'height': 0.18, 'color': '#0284c7', 'border': '#38bdf8'
        },
        {
            'title': 'LAYER 2: PEARL LEVEL-3 STRUCTURAL CAUSAL DIGITAL TWIN (Python SIMD / CSR)',
            'desc': 'Abduction u_i* = y_i - f(x_i,z_i)  |  Action do(R_i=0.95)  |  Prediction y_i^do  |  Sparse CSR Random-Walk Matrix P_f',
            'xy': (0.05, 0.53), 'width': 0.90, 'height': 0.18, 'color': '#ea580c', 'border': '#f97316'
        },
        {
            'title': 'LAYER 3: PLECKO SFM CAUSAL MEDIATION ENGINE (R / faircause / dagitty)',
            'desc': 'Ctf-DE (21.4%)  |  Ctf-IE_R (61.3% Hardware Attribution)  |  Ctf-IE_T (-17.3% Topology Buffer)  |  1,000-Sample Bootstrap 95% CIs',
            'xy': (0.05, 0.29), 'width': 0.90, 'height': 0.18, 'color': '#7c3aed', 'border': '#a855f7'
        },
        {
            'title': 'LAYER 4: GRAPH LAPLACIAN & PARETO DOMINANCE ENGINE (R & Python)',
            'desc': 'L_sym = I - D^(-1/2) A D^(-1/2) (13 Spectral Clusters)  |  Pareto Dominance Frontier (Level-3 Repair Dominates FairSTG Software)',
            'xy': (0.05, 0.05), 'width': 0.90, 'height': 0.18, 'color': '#059669', 'border': '#34d399'
        }
    ]
    
    for layer in layers:
        # Add Rectangle Box
        rect = patches.FancyBboxPatch(
            layer['xy'], layer['width'], layer['height'],
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=layer['color'], edgecolor=layer['border'],
            linewidth=2.5, alpha=0.9, transform=ax.transAxes
        )
        ax.add_patch(rect)
        
        # Add Layer Title
        ax.text(
            layer['xy'][0] + 0.03, layer['xy'][1] + 0.12,
            layer['title'], fontsize=11, fontweight='bold',
            color='#ffffff', transform=ax.transAxes
        )
        
        # Add Layer Description
        ax.text(
            layer['xy'][0] + 0.03, layer['xy'][1] + 0.04,
            layer['desc'], fontsize=9, fontweight='medium',
            color='#f1f5f9', transform=ax.transAxes
        )
        
    # Draw Vertical Connecting Arrows between Layers
    arrow_props = dict(facecolor='#94a3b8', edgecolor='#e2e8f0', width=3, headwidth=10, headlength=8)
    
    # Arrow 1 -> 2
    ax.annotate('', xy=(0.50, 0.72), xytext=(0.50, 0.77), xycoords='axes fraction', arrowprops=arrow_props)
    ax.text(0.52, 0.74, "Telemetry & Interventions", color="#38bdf8", fontsize=8, fontweight='bold', transform=ax.transAxes)
    
    # Arrow 2 -> 3
    ax.annotate('', xy=(0.50, 0.48), xytext=(0.50, 0.53), xycoords='axes fraction', arrowprops=arrow_props)
    ax.text(0.52, 0.50, "Counterfactual Outcomes", color="#f97316", fontsize=8, fontweight='bold', transform=ax.transAxes)
    
    # Arrow 3 -> 4
    ax.annotate('', xy=(0.50, 0.24), xytext=(0.50, 0.29), xycoords='axes fraction', arrowprops=arrow_props)
    ax.text(0.52, 0.26, "District Cluster DAGs", color="#a855f7", fontsize=8, fontweight='bold', transform=ax.transAxes)
    
    plt.title("METR-LA Structural Causal Digital Twin — Polyglot Multi-Layer Architecture\n[Pearl Level-3 do-Calculus, Plecko Structural Fairness Model & Spectral Graph Laplacian]", 
              fontsize=12, fontweight='bold', color='#38bdf8', pad=25)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save Diagram
    out_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(out_dir, "system_architecture_publication_diagram.png")
    plt.savefig(img_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"[OK] Publication-grade System Architecture diagram generated: {img_path}")
    
    # Copy to root directory if called from subdirectory
    root_copy = os.path.join(out_dir, "..", "..", "system_architecture_publication_diagram.png")
    if os.path.exists(os.path.dirname(root_copy)):
        shutil.copy(img_path, root_copy)

if __name__ == '__main__':
    main()
