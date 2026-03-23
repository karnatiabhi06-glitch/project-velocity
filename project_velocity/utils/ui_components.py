
import streamlit.components.v1 as components

def particles_background(color="#ffffff", bg_color="#000000", quantity=100):
    """
    Renders a full-screen particle background using vanilla JS in an iframe.
    Args:
        color: Particle color
        bg_color: Background color for the page (so iframe acts as the background)
        quantity: Number of particles
    """
    
    js_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body, html {{
                margin: 0;
                padding: 0;
                width: 100vw;
                height: 100vh;
                background-color: {bg_color} !important;
                overflow: hidden;
            }}
            canvas {{
                display: block;
                position: absolute;
                top: 0;
                left: 0;
            }}
        </style>
    </head>
    <body>
        <canvas id="canvas"></canvas>
        <script>
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            
            let width, height;
            let particles = [];
            const particleColor = "{color}";
            const particleQuantity = {quantity};
            
            function resize() {{
                width = window.innerWidth;
                height = window.innerHeight;
                canvas.width = width;
                canvas.height = height;
                initParticles();
            }}
            
            function initParticles() {{
                particles = [];
                for (let i = 0; i < particleQuantity; i++) {{
                    particles.push({{
                        x: Math.random() * width,
                        y: Math.random() * height,
                        vx: (Math.random() - 0.5) * 0.5,
                        vy: (Math.random() - 0.5) * 0.5,
                        size: Math.random() * 2 + 0.5,
                        alpha: Math.random() * 0.5 + 0.2
                    }});
                }}
            }}
            
            function animate() {{
                ctx.clearRect(0, 0, width, height);
                ctx.fillStyle = particleColor;
                
                particles.forEach(p => {{
                    p.x += p.vx;
                    p.y += p.vy;
                    
                    // Wrap around
                    if (p.x < 0) p.x = width;
                    if (p.x > width) p.x = 0;
                    if (p.y < 0) p.y = height;
                    if (p.y > height) p.y = 0;
                    
                    ctx.globalAlpha = p.alpha;
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                    ctx.fill();
                }});
                requestAnimationFrame(animate);
            }}
            
            window.addEventListener('resize', resize);
            resize();
            animate();
        </script>
    </body>
    </html>
    """
    
    # Use a specific height to target this iframe via CSS.
    components.html(js_html, height=101, scrolling=False)

