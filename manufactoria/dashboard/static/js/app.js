// Manufactoria Problem Manager - JavaScript utilities

// Global utilities
const Utils = {
    // Show loading state on buttons
    setButtonLoading: function(button, isLoading, loadingText = 'Loading...') {
        if (isLoading) {
            button.dataset.originalText = button.innerHTML;
            button.innerHTML = `<i class="fas fa-spinner fa-spin me-1"></i>${loadingText}`;
            button.disabled = true;
        } else {
            button.innerHTML = button.dataset.originalText || button.innerHTML;
            button.disabled = false;
        }
    },

    // Show toast notifications
    showToast: function(message, type = 'info') {
        const toastContainer = this.getOrCreateToastContainer();
        const toastId = 'toast_' + Date.now();
        
        const toastHtml = `
            <div id="${toastId}" class="toast align-items-center text-white bg-${type} border-0" role="alert">
                <div class="d-flex">
                    <div class="toast-body">
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;
        
        toastContainer.insertAdjacentHTML('beforeend', toastHtml);
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement);
        
        toast.show();
        
        // Auto-remove after hiding
        toastElement.addEventListener('hidden.bs.toast', function() {
            toastElement.remove();
        });
    },

    // Get or create toast container
    getOrCreateToastContainer: function() {
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container position-fixed top-0 end-0 p-3';
            container.style.zIndex = '1050';
            document.body.appendChild(container);
        }
        return container;
    },

    // Format dates nicely
    formatDate: function(dateString) {
        if (!dateString) return 'Unknown';
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    },

    // Validate DSL code format
    validateDslFormat: function(dslCode) {
        const lines = dslCode.trim().split('\n');
        const errors = [];
        
        if (!dslCode.trim()) {
            errors.push('DSL code cannot be empty');
            return errors;
        }
        
        // Check for START node
        const hasStart = lines.some(line => line.trim().startsWith('START '));
        if (!hasStart) {
            errors.push('DSL must contain a START node');
        }
        
        // Check for END node
        const hasEnd = lines.some(line => line.trim().startsWith('END '));
        if (!hasEnd) {
            errors.push('DSL must contain an END node');
        }
        
        return errors;
    },

    // Debounce function for search/input
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};

// API wrapper functions
const API = {
    // Generic API call wrapper
    call: async function(url, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json'
            }
        };
        
        const mergedOptions = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(url, mergedOptions);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }
            
            return data;
        } catch (error) {
            console.error('API call failed:', error);
            throw error;
        }
    },

    // Get all problems
    getProblems: function() {
        return this.call('/api/problems');
    },

    // Create a new problem
    createProblem: function(problemData) {
        return this.call('/api/problems', {
            method: 'POST',
            body: JSON.stringify(problemData)
        });
    },

    // Update a problem
    updateProblem: function(problemId, problemData) {
        return this.call(`/api/problems/${problemId}`, {
            method: 'PUT',
            body: JSON.stringify(problemData)
        });
    },

    // Delete a problem
    deleteProblem: function(problemId) {
        return this.call(`/api/problems/${problemId}`, {
            method: 'DELETE'
        });
    },

    // Validate DSL code
    validateDsl: function(dslCode) {
        return this.call('/api/validate_dsl', {
            method: 'POST',
            body: JSON.stringify({ dsl: dslCode })
        });
    },

    // Test solution against test cases
    testSolution: function(dslCode, testCases) {
        return this.call('/api/test_solution', {
            method: 'POST',
            body: JSON.stringify({
                dsl: dslCode,
                test_cases: testCases
            })
        });
    }
};

// DSL Editor enhancements
const DSLEditor = {
    // Initialize code editors with basic syntax highlighting
    init: function() {
        const textareas = document.querySelectorAll('.code-editor');
        textareas.forEach(textarea => {
            this.enhanceTextarea(textarea);
        });
    },

    // Enhance textarea with basic features
    enhanceTextarea: function(textarea) {
        // Add line numbers and basic formatting
        textarea.addEventListener('keydown', function(e) {
            // Tab key support
            if (e.key === 'Tab') {
                e.preventDefault();
                const start = this.selectionStart;
                const end = this.selectionEnd;
                
                // Insert 4 spaces
                this.value = this.value.substring(0, start) + '    ' + this.value.substring(end);
                this.selectionStart = this.selectionEnd = start + 4;
            }
            
            // Auto-indent on Enter
            if (e.key === 'Enter') {
                const start = this.selectionStart;
                const lines = this.value.substring(0, start).split('\n');
                const currentLine = lines[lines.length - 1];
                const indent = currentLine.match(/^\s*/)[0];
                
                // Add extra indent for node definitions
                const extraIndent = currentLine.includes(':') ? '    ' : '';
                
                setTimeout(() => {
                    const newStart = this.selectionStart;
                    this.value = this.value.substring(0, newStart) + indent + extraIndent + this.value.substring(newStart);
                    this.selectionStart = this.selectionEnd = newStart + indent.length + extraIndent.length;
                }, 0);
            }
        });

        // Add basic syntax validation on blur
        textarea.addEventListener('blur', function() {
            const errors = Utils.validateDslFormat(this.value);
            const feedback = this.parentNode.querySelector('.dsl-feedback');
            
            if (feedback) {
                feedback.remove();
            }
            
            if (errors.length > 0) {
                const feedbackDiv = document.createElement('div');
                feedbackDiv.className = 'dsl-feedback text-danger small mt-1';
                feedbackDiv.innerHTML = '<i class="fas fa-exclamation-triangle me-1"></i>' + errors.join(', ');
                this.parentNode.appendChild(feedbackDiv);
            }
        });
    }
};

// Form validation utilities
const FormValidator = {
    // Validate problem form
    validateProblemForm: function(formData) {
        const errors = [];
        
        if (!formData.name || formData.name.trim().length === 0) {
            errors.push('Problem name is required');
        }
        
        if (formData.available_nodes.length === 0) {
            errors.push('At least one node type must be selected');
        }
        
        // Validate that START and END are always included
        if (!formData.available_nodes.includes('START')) {
            errors.push('START node type must be selected');
        }
        
        if (!formData.available_nodes.includes('END')) {
            errors.push('END node type must be selected');
        }
        
        return errors;
    },

    // Show form errors
    showFormErrors: function(errors) {
        // Remove existing error display
        const existingAlert = document.querySelector('.form-errors');
        if (existingAlert) {
            existingAlert.remove();
        }
        
        if (errors.length === 0) return;
        
        const errorHtml = `
            <div class="alert alert-danger form-errors">
                <h6><i class="fas fa-exclamation-triangle me-2"></i>Please fix the following errors:</h6>
                <ul class="mb-0">
                    ${errors.map(error => `<li>${error}</li>`).join('')}
                </ul>
            </div>
        `;
        
        const form = document.getElementById('problemForm');
        if (form) {
            form.insertAdjacentHTML('afterbegin', errorHtml);
            form.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }
};

// Search and filter functionality
const SearchFilter = {
    // Initialize search functionality
    init: function() {
        const searchInput = document.getElementById('problemSearch');
        if (searchInput) {
            searchInput.addEventListener('input', Utils.debounce(this.filterProblems.bind(this), 300));
        }
        
        const filterButtons = document.querySelectorAll('.filter-btn');
        filterButtons.forEach(btn => {
            btn.addEventListener('click', this.handleFilterClick.bind(this));
        });
    },

    // Filter problems based on search term
    filterProblems: function(searchTerm) {
        const problemCards = document.querySelectorAll('.problem-card');
        const term = (searchTerm.target?.value || searchTerm).toLowerCase();
        
        problemCards.forEach(card => {
            const title = card.querySelector('.card-title').textContent.toLowerCase();
            const description = card.querySelector('.card-text').textContent.toLowerCase();
            const matches = title.includes(term) || description.includes(term);
            
            card.closest('.col-md-6').style.display = matches ? 'block' : 'none';
        });
        
        this.updateNoResultsMessage();
    },

    // Handle filter button clicks
    handleFilterClick: function(e) {
        const filter = e.target.dataset.filter;
        const problemCards = document.querySelectorAll('.problem-card');
        
        // Update active filter button
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        e.target.classList.add('active');
        
        problemCards.forEach(card => {
            const badges = card.querySelectorAll('.badge');
            let show = filter === 'all';
            
            if (filter === 'with-tests') {
                show = Array.from(badges).some(badge => badge.textContent.includes('tests'));
            } else if (filter === 'with-solutions') {
                show = Array.from(badges).some(badge => badge.textContent.includes('solutions'));
            }
            
            card.closest('.col-md-6').style.display = show ? 'block' : 'none';
        });
        
        this.updateNoResultsMessage();
    },

    // Update no results message
    updateNoResultsMessage: function() {
        const visibleCards = document.querySelectorAll('.col-md-6[style="display: block;"], .col-md-6:not([style*="display: none"])');
        const noResults = document.getElementById('noResults');
        
        if (visibleCards.length === 0 && !noResults) {
            const container = document.querySelector('.row');
            if (container) {
                const noResultsHtml = `
                    <div id="noResults" class="col-12 text-center py-5">
                        <i class="fas fa-search text-muted" style="font-size: 3rem;"></i>
                        <h5 class="text-muted mt-3">No problems found</h5>
                        <p class="text-muted">Try adjusting your search or filter criteria</p>
                    </div>
                `;
                container.insertAdjacentHTML('beforeend', noResultsHtml);
            }
        } else if (visibleCards.length > 0 && noResults) {
            noResults.remove();
        }
    }
};

// Auto-save functionality for forms
const AutoSave = {
    // Initialize auto-save
    init: function() {
        const form = document.getElementById('problemForm');
        if (form) {
            this.setupAutoSave(form);
        }
    },

    // Setup auto-save for a form
    setupAutoSave: function(form) {
        const inputs = form.querySelectorAll('input, textarea, select');
        const saveKey = 'autosave_' + window.location.pathname + '_' + Date.now();
        
        // Don't load saved data for edit forms to prevent data corruption
        // this.loadSavedData(saveKey, inputs);
        
        // Save on input
        inputs.forEach(input => {
            input.addEventListener('input', Utils.debounce(() => {
                this.saveFormData(saveKey, inputs);
            }, 1000));
        });
        
        // Clear saved data on successful submit
        form.addEventListener('submit', () => {
            localStorage.removeItem(saveKey);
        });
    },

    // Save form data to localStorage
    saveFormData: function(key, inputs) {
        const data = {};
        inputs.forEach(input => {
            if (input.type === 'checkbox') {
                data[input.name] = input.checked;
            } else if (input.type === 'radio') {
                if (input.checked) {
                    data[input.name] = input.value;
                }
            } else {
                data[input.name] = input.value;
            }
        });
        
        localStorage.setItem(key, JSON.stringify(data));
    },

    // Load saved data from localStorage
    loadSavedData: function(key, inputs) {
        const savedData = localStorage.getItem(key);
        if (!savedData) return;
        
        try {
            const data = JSON.parse(savedData);
            inputs.forEach(input => {
                if (data.hasOwnProperty(input.name)) {
                    if (input.type === 'checkbox') {
                        input.checked = data[input.name];
                    } else if (input.type === 'radio') {
                        input.checked = input.value === data[input.name];
                    } else {
                        input.value = data[input.name];
                    }
                }
            });
        } catch (error) {
            console.warn('Failed to load auto-saved data:', error);
        }
    }
};

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Clear any problematic auto-save data for edit forms
    if (window.location.pathname.includes('/edit/')) {
        const problemId = window.location.pathname.split('/edit/')[1];
        const keysToRemove = [];
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key && key.startsWith('autosave_')) {
                keysToRemove.push(key);
            }
        }
        keysToRemove.forEach(key => localStorage.removeItem(key));
    }
    
    // Initialize all modules
    DSLEditor.init();
    SearchFilter.init();
    AutoSave.init();
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + S to save (prevent default and trigger form submit if on edit page)
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            const form = document.getElementById('problemForm');
            if (form) {
                e.preventDefault();
                form.dispatchEvent(new Event('submit'));
            }
        }
        
        // Ctrl/Cmd + N for new problem
        if ((e.ctrlKey || e.metaKey) && e.key === 'n' && window.location.pathname === '/') {
            e.preventDefault();
            window.location.href = '/new';
        }
    });
});

// ===============================================
// VISUAL FACTORY BUILDER
// ===============================================

// Global registry for visual builder instances
window.visualBuilderInstances = {};

const VisualFactoryBuilder = {
    // Create a new instance of the visual builder
    createInstance: function(containerId, availableNodes = []) {
        const instance = Object.create(this);
        instance.init(containerId, availableNodes);
        window.visualBuilderInstances[containerId] = instance;
        return instance;
    },
    
    // Get existing instance
    getInstance: function(containerId) {
        return window.visualBuilderInstances[containerId];
    },
    
    // Initialize the visual builder
    init: function(containerId, availableNodes = []) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error('Visual factory builder container not found');
            return;
        }
        
        this.containerId = containerId;
        this.availableNodes = availableNodes.length > 0 ? availableNodes : [
            'START', 'END', 'PULLER_RB', 'PULLER_YG', 
            'PAINTER_RED', 'PAINTER_BLUE', 'PAINTER_YELLOW', 'PAINTER_GREEN'
        ];
        
        this.nodes = new Map();
        this.connections = new Map();
        this.nodeCounter = 0;
        this.selectedNode = null;
        this.connectionMode = false;
        this.connectionStart = null;
        this.dragOffset = { x: 0, y: 0 };
        this.isDragging = false;
        this.draggedNode = null;
        this.zoomLevel = 1;
        this.minZoom = 0.25;
        this.maxZoom = 3;
        this.zoomStep = 0.25;
        
        this.setupBuilder();
        this.setupEventListeners();
    },
    
    // Setup the visual builder HTML structure
    setupBuilder: function() {
        this.container.innerHTML = `
            <div class="visual-builder">
                <!-- Node Palette -->
                <div class="node-palette">
                    <h6>Node Types</h6>
                    <div class="palette-nodes">
                        ${this.availableNodes.map(nodeType => `
                            <div class="palette-node" 
                                 data-node-type="${nodeType}" 
                                 draggable="true"
                                 title="Drag to workspace to create ${nodeType} node">
                                ${this.getNodeDisplayName(nodeType)}
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                <!-- Builder Toolbar -->
                <div class="builder-toolbar">
                    <button type="button" class="btn btn-sm btn-outline-secondary clear-btn" title="Clear All">
                        <i class="fas fa-trash"></i>
                    </button>
                    <button type="button" class="btn btn-sm btn-outline-primary center-btn" title="Center View">
                        <i class="fas fa-crosshairs"></i>
                    </button>
                    <button type="button" class="btn btn-sm btn-outline-info generate-dsl-btn" title="Generate DSL">
                        <i class="fas fa-code"></i>
                    </button>
                    <button type="button" class="btn btn-sm btn-outline-success auto-layout-btn" title="Auto-arrange Nodes">
                        <i class="fas fa-sitemap"></i>
                    </button>
                    <button type="button" class="btn btn-sm btn-outline-dark zoom-out-btn" title="Zoom Out">
                        <i class="fas fa-search-minus"></i>
                    </button>
                    <button type="button" class="btn btn-sm btn-outline-dark zoom-in-btn" title="Zoom In">
                        <i class="fas fa-search-plus"></i>
                    </button>
                    <button type="button" class="btn btn-sm btn-outline-warning help-btn" title="How to Connect Nodes">
                        <i class="fas fa-question"></i>
                    </button>
                </div>
                
                <!-- Workspace -->
                <div class="builder-workspace" id="builderWorkspace">
                    <div class="workspace-container">
                        <!-- SVG for connections -->
                        <svg class="connection-svg">
                            <defs>
                                <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                                        refX="10" refY="3.5" orient="auto">
                                    <polygon points="0 0, 10 3.5, 0 7" fill="#6c757d" />
                                </marker>
                            </defs>
                        </svg>
                    </div>
                </div>
            </div>
        `;
        
        this.workspace = this.container.querySelector('.builder-workspace');
        this.workspaceContainer = this.container.querySelector('.workspace-container');
        this.svg = this.container.querySelector('.connection-svg');
        
        // Initialize zoom state
        this.applyZoom();
    },
    
    // Setup event listeners
    setupEventListeners: function() {
        // Palette node drag events
        this.container.addEventListener('dragstart', this.handlePaletteDragStart.bind(this));
        this.container.addEventListener('dragend', this.handlePaletteDragEnd.bind(this));
        
        // Workspace drop events
        this.workspace.addEventListener('dragover', this.handleWorkspaceDragOver.bind(this));
        this.workspace.addEventListener('drop', this.handleWorkspaceDrop.bind(this));
        
        // Workspace click events
        this.workspace.addEventListener('click', this.handleWorkspaceClick.bind(this));
        this.workspace.addEventListener('mousedown', this.handleWorkspaceMouseDown.bind(this));
        this.workspace.addEventListener('mousemove', this.handleWorkspaceMouseMove.bind(this));
        this.workspace.addEventListener('mouseup', this.handleWorkspaceMouseUp.bind(this));
        
        // Toolbar button events
        const clearBtn = this.container.querySelector('.clear-btn');
        const centerBtn = this.container.querySelector('.center-btn');
        const generateBtn = this.container.querySelector('.generate-dsl-btn');
        const autoLayoutBtn = this.container.querySelector('.auto-layout-btn');
        const zoomInBtn = this.container.querySelector('.zoom-in-btn');
        const zoomOutBtn = this.container.querySelector('.zoom-out-btn');
        const helpBtn = this.container.querySelector('.help-btn');
        
        if (clearBtn) clearBtn.addEventListener('click', this.clearWorkspace.bind(this));
        if (centerBtn) centerBtn.addEventListener('click', this.centerView.bind(this));
        if (generateBtn) generateBtn.addEventListener('click', () => {
            const dsl = this.generateDSL();
            if (dsl) {
                alert('Generated DSL:\n\n' + dsl);
            } else {
                alert('No nodes to generate DSL from');
            }
        });
        if (autoLayoutBtn) autoLayoutBtn.addEventListener('click', () => {
            if (this.nodes.size === 0) {
                Utils.showToast('No nodes to arrange', 'warning');
                return;
            }
            this.applyIntelligentLayout();
            this.updateNodePositions();
            this.updateConnections();
            Utils.showToast('Nodes automatically arranged', 'success');
        });
        if (zoomInBtn) zoomInBtn.addEventListener('click', this.zoomIn.bind(this));
        if (zoomOutBtn) zoomOutBtn.addEventListener('click', this.zoomOut.bind(this));
        if (helpBtn) helpBtn.addEventListener('click', () => {
            alert('How to Use the Visual Factory Builder:\n\n' +
                  'Building Nodes:\n' +
                  '1. Drag node types from the palette to the workspace\n' +
                  '2. Edit node IDs by clicking on the text field\n' +
                  '3. Delete nodes using the X button\n\n' +
                  'Connecting Nodes:\n' +
                  '4. For PULLER nodes: Click the colored output circles (red/blue or yellow/green) or gray EMPTY circle\n' +
                  '5. For START/PAINTER nodes: Click the blue output circle\n' +
                  '6. Click the blue input circle (●) on the LEFT side of another node to complete the connection\n' +
                  '7. Click on connection lines to delete them\n\n' +
                  'Navigation:\n' +
                  '8. Use zoom buttons or Ctrl/Cmd + +/- to zoom in/out\n' +
                  '9. Use Ctrl/Cmd + 0 to reset zoom to 100%\n' +
                  '10. Use center button to center all nodes in view\n\n' +
                  'Color Guide: Red=R, Blue=B, Yellow=Y, Green=G, Gray=EMPTY, Blue=NEXT');
        });
        
        // Global events
        document.addEventListener('keydown', this.handleKeyDown.bind(this));
    },
    
    // Get display name for node types
    getNodeDisplayName: function(nodeType) {
        const names = {
            'START': 'START',
            'END': 'END',
            'PULLER_RB': 'P_RB',
            'PULLER_YG': 'P_YG',
            'PAINTER_RED': 'P_R',
            'PAINTER_BLUE': 'P_B',
            'PAINTER_YELLOW': 'P_Y',
            'PAINTER_GREEN': 'P_G'
        };
        return names[nodeType] || nodeType;
    },
    
    // Handle palette node drag start
    handlePaletteDragStart: function(e) {
        if (e.target.classList.contains('palette-node')) {
            const nodeType = e.target.dataset.nodeType;
            e.dataTransfer.setData('text/plain', nodeType);
            e.dataTransfer.effectAllowed = 'copy';
            
            // Create drag preview
            const preview = e.target.cloneNode(true);
            preview.classList.add('drag-preview');
            document.body.appendChild(preview);
            e.dataTransfer.setDragImage(preview, 20, 15);
            
            // Clean up preview after drag
            setTimeout(() => document.body.removeChild(preview), 0);
        }
    },
    
    // Handle palette node drag end
    handlePaletteDragEnd: function(e) {
        // Cleanup any drag state
    },
    
    // Handle workspace drag over
    handleWorkspaceDragOver: function(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    },
    
    // Handle workspace drop
    handleWorkspaceDrop: function(e) {
        e.preventDefault();
        const nodeType = e.dataTransfer.getData('text/plain');
        if (nodeType && this.availableNodes.includes(nodeType)) {
            const rect = this.workspaceContainer.getBoundingClientRect();
            const x = (e.clientX - rect.left) / this.zoomLevel;
            const y = (e.clientY - rect.top) / this.zoomLevel;
            this.createNode(nodeType, x, y);
        }
    },
    
    // Create a new node
    createNode: function(nodeType, x, y) {
        // Generate unique node ID
        const baseId = nodeType === 'START' ? 'start' : 
                      nodeType === 'END' ? 'end' : 
                      nodeType.toLowerCase().replace('_', '') + (++this.nodeCounter);
        
        // Check for existing START/END nodes
        if (nodeType === 'START' && Array.from(this.nodes.values()).some(n => n.type === 'START')) {
            Utils.showToast('Only one START node is allowed', 'warning');
            return;
        }
        if (nodeType === 'END' && Array.from(this.nodes.values()).some(n => n.type === 'END')) {
            Utils.showToast('Only one END node is allowed', 'warning');
            return;
        }
        
        const nodeId = this.generateUniqueId(baseId);
        
        const node = {
            id: nodeId,
            type: nodeType,
            x: x - 40, // Center the node on drop point
            y: y - 25,
            routes: []
        };
        
        this.nodes.set(nodeId, node);
        this.renderNode(node);
        this.generateDSL();
    },
    
    // Generate unique ID
    generateUniqueId: function(baseId) {
        let id = baseId;
        let counter = 1;
        while (this.nodes.has(id)) {
            id = baseId + counter;
            counter++;
        }
        return id;
    },
    
    // Render a node in the workspace
    renderNode: function(node) {
        const nodeElement = document.createElement('div');
        nodeElement.className = 'factory-node';
        nodeElement.dataset.nodeType = node.type;
        nodeElement.dataset.nodeId = node.id;
        nodeElement.style.left = node.x + 'px';
        nodeElement.style.top = node.y + 'px';
        
        nodeElement.innerHTML = `
            <div class="node-type">${node.type}</div>
            <input type="text" class="node-id" value="${node.id}">
            <div class="delete-btn">&times;</div>
            ${this.getOutputConnectionsHTML(node.type)}
            ${this.needsInputConnection(node.type) ? '<div class="connection-point input"></div>' : ''}
        `;
        
        this.workspaceContainer.appendChild(nodeElement);
        
        // Setup event listeners for node elements
        const nodeIdInput = nodeElement.querySelector('.node-id');
        const deleteBtn = nodeElement.querySelector('.delete-btn');
        const outputPoints = nodeElement.querySelectorAll('.connection-point.output');
        const inputPoint = nodeElement.querySelector('.connection-point.input');
        
        if (nodeIdInput) {
            nodeIdInput.addEventListener('change', (e) => {
                this.updateNodeId(node.id, e.target.value);
            });
        }
        
        if (deleteBtn) {
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteNode(node.id);
            });
        }
        
        // Handle multiple output points
        outputPoints.forEach(outputPoint => {
            outputPoint.addEventListener('click', (e) => {
                e.stopPropagation();
                this.startConnection(outputPoint);
            });
        });
        
        if (inputPoint) {
            inputPoint.addEventListener('click', (e) => {
                e.stopPropagation();
                this.completeConnection(inputPoint);
            });
        }
        
        // Make node draggable
        this.makeNodeDraggable(nodeElement);
    },
    
    // Check if node type needs output connection
    needsOutputConnection: function(nodeType) {
        return nodeType !== 'END';
    },
    
    // Get HTML for output connection points based on node type
    getOutputConnectionsHTML: function(nodeType) {
        if (nodeType === 'END') {
            return '';
        } else if (nodeType === 'PULLER_RB') {
            return `
                <div class="connection-point output output-1" data-condition="R" title="Red output"></div>
                <div class="connection-point output output-2" data-condition="B" title="Blue output"></div>
                <div class="connection-point output output-3" data-condition="EMPTY" title="Empty output"></div>
            `;
        } else if (nodeType === 'PULLER_YG') {
            return `
                <div class="connection-point output output-1" data-condition="Y" title="Yellow output"></div>
                <div class="connection-point output output-2" data-condition="G" title="Green output"></div>
                <div class="connection-point output output-3" data-condition="EMPTY" title="Empty output"></div>
            `;
        } else {
            // START and PAINTER nodes have single output (NEXT)
            return '<div class="connection-point output" data-condition="NEXT" title="Next output"></div>';
        }
    },
    
    // Check if node type needs input connection
    needsInputConnection: function(nodeType) {
        return nodeType !== 'START';
    },
    
    // Get height for node type
    getNodeHeight: function(nodeType) {
        if (nodeType === 'PULLER_RB' || nodeType === 'PULLER_YG') {
            return 80; // Increased height for puller nodes
        }
        return 50; // Default height for other nodes
    },
    
    // Find output point by condition
    findOutputPointByCondition: function(nodeId, condition) {
        const nodeElement = this.workspaceContainer.querySelector(`[data-node-id="${nodeId}"]`);
        if (!nodeElement) return null;
        
        // For NEXT or null condition, find the basic output point
        if (!condition || condition === 'NEXT') {
            return nodeElement.querySelector('.connection-point.output:not(.output-1):not(.output-2):not(.output-3)');
        }
        
        // For specific conditions, find the matching output point
        return nodeElement.querySelector(`.connection-point.output[data-condition="${condition}"]`);
    },
    
    // Apply intelligent layout for loaded DSL
    applyIntelligentLayout: function() {
        if (this.nodes.size === 0) return;
        
        // Find START and END nodes
        const startNode = Array.from(this.nodes.values()).find(n => n.type === 'START');
        const endNode = Array.from(this.nodes.values()).find(n => n.type === 'END');
        
        if (!startNode) return;
        
        // Create a map of node levels (distance from START)
        const nodeLevels = new Map();
        const visited = new Set();
        
        // BFS to assign levels
        const queue = [{ node: startNode, level: 0 }];
        nodeLevels.set(startNode.id, 0);
        visited.add(startNode.id);
        
        while (queue.length > 0) {
            const { node, level } = queue.shift();
            
            // Add connected nodes to next level
            node.routes.forEach(route => {
                if (this.nodes.has(route.target) && !visited.has(route.target)) {
                    const targetNode = this.nodes.get(route.target);
                    nodeLevels.set(route.target, level + 1);
                    visited.add(route.target);
                    queue.push({ node: targetNode, level: level + 1 });
                }
            });
        }
        
        // Handle nodes not reachable from START
        this.nodes.forEach(node => {
            if (!nodeLevels.has(node.id)) {
                if (node.type === 'END') {
                    // Place END at a high level if not connected
                    const maxLevel = Math.max(...Array.from(nodeLevels.values()), 0);
                    nodeLevels.set(node.id, maxLevel + 1);
                } else {
                    // Place orphaned nodes at level 1
                    nodeLevels.set(node.id, 1);
                }
            }
        });
        
        // Group nodes by level
        const levels = new Map();
        nodeLevels.forEach((level, nodeId) => {
            if (!levels.has(level)) {
                levels.set(level, []);
            }
            levels.get(level).push(this.nodes.get(nodeId));
        });
        
        // Layout parameters
        const startX = 150;
        const startY = 200;
        const levelSpacing = 300;
        const nodeSpacing = 150;
        
        // Position nodes level by level
        const sortedLevels = Array.from(levels.keys()).sort((a, b) => a - b);
        
        sortedLevels.forEach(levelIndex => {
            const nodesInLevel = levels.get(levelIndex);
            const levelX = startX + levelIndex * levelSpacing;
            
            // Sort nodes in level: START first, END last, others in between
            nodesInLevel.sort((a, b) => {
                if (a.type === 'START') return -1;
                if (b.type === 'START') return 1;
                if (a.type === 'END') return 1;
                if (b.type === 'END') return -1;
                return a.id.localeCompare(b.id);
            });
            
            // Position nodes vertically within the level
            const totalHeight = (nodesInLevel.length - 1) * nodeSpacing;
            const levelStartY = startY - totalHeight / 2;
            
            nodesInLevel.forEach((node, index) => {
                node.x = levelX;
                node.y = levelStartY + index * nodeSpacing;
            });
        });
        
        // Special positioning for END node - ensure it's rightmost
        if (endNode) {
            const maxX = Math.max(...Array.from(this.nodes.values()).map(n => n.x));
            if (endNode.x < maxX) {
                endNode.x = maxX + levelSpacing;
            }
        }
    },
    
    // Make node draggable
    makeNodeDraggable: function(nodeElement) {
        let isDragging = false;
        let startX, startY, nodeStartX, nodeStartY;
        
        nodeElement.addEventListener('mousedown', (e) => {
            if (e.target.classList.contains('connection-point') || 
                e.target.classList.contains('delete-btn') ||
                e.target.classList.contains('node-id')) {
                return;
            }
            
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            nodeStartX = parseInt(nodeElement.style.left);
            nodeStartY = parseInt(nodeElement.style.top);
            
            nodeElement.style.zIndex = '200';
            e.preventDefault();
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            
            // Account for zoom level in drag calculations
            const deltaX = (e.clientX - startX) / this.zoomLevel;
            const deltaY = (e.clientY - startY) / this.zoomLevel;
            const newX = nodeStartX + deltaX;
            const newY = nodeStartY + deltaY;
            
            nodeElement.style.left = Math.max(0, newX) + 'px';
            nodeElement.style.top = Math.max(0, newY) + 'px';
            
            // Update node data
            const nodeId = nodeElement.dataset.nodeId;
            const node = this.nodes.get(nodeId);
            if (node) {
                node.x = newX;
                node.y = newY;
            }
            
            this.updateConnections();
        });
        
        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                nodeElement.style.zIndex = '100';
            }
        });
    },
    
    // Start connection from a node
    startConnection: function(connectionPoint) {
        const nodeElement = connectionPoint.closest('.factory-node');
        const nodeId = nodeElement.dataset.nodeId;
        const condition = connectionPoint.dataset.condition;
        
        // If already in connection mode, cancel it
        if (this.connectionMode) {
            this.connectionMode = false;
            this.connectionStart = null;
            this.connectionStartCondition = null;
            this.connectionStartPoint = null;
            this.clearConnectionHighlights();
            Utils.showToast('Connection cancelled', 'info');
            return;
        }
        
        // Start connection mode
        this.connectionMode = true;
        this.connectionStart = nodeId;
        this.connectionStartCondition = condition;
        this.connectionStartPoint = connectionPoint;
        this.highlightConnectionTargets(nodeId);
        
        // Highlight the starting node
        nodeElement.classList.add('connecting');
        
        const conditionText = condition === 'NEXT' ? '' : `[${condition}] `;
        Utils.showToast(`Starting ${conditionText}connection - click on an input point (●) to complete`, 'info');
    },
    
    // Complete connection to a node
    completeConnection: function(connectionPoint) {
        if (!this.connectionMode || !this.connectionStart) {
            Utils.showToast('No connection in progress', 'warning');
            return;
        }
        
        const nodeElement = connectionPoint.closest('.factory-node');
        const nodeId = nodeElement.dataset.nodeId;
        
        // Complete the connection with the predetermined condition
        this.createConnection(this.connectionStart, nodeId, this.connectionStartCondition, this.connectionStartPoint);
        this.connectionMode = false;
        this.connectionStart = null;
        this.connectionStartCondition = null;
        this.connectionStartPoint = null;
        this.clearConnectionHighlights();
    },
    
    // Highlight valid connection targets
    highlightConnectionTargets: function(fromNodeId) {
        this.clearConnectionHighlights();
        
        const fromNode = this.nodes.get(fromNodeId);
        this.workspaceContainer.querySelectorAll('.factory-node').forEach(nodeEl => {
            const nodeId = nodeEl.dataset.nodeId;
            // Allow self-connections or connections to nodes that need input
            if (nodeId === fromNodeId || this.needsInputConnection(this.nodes.get(nodeId).type)) {
                nodeEl.classList.add('connecting');
                
                // Also highlight the input connection point specifically
                const inputPoint = nodeEl.querySelector('.connection-point.input');
                if (inputPoint) {
                    inputPoint.classList.add('highlighted');
                }
            }
        });
    },
    
    // Clear connection highlights
    clearConnectionHighlights: function() {
        this.workspaceContainer.querySelectorAll('.factory-node').forEach(nodeEl => {
            nodeEl.classList.remove('connecting');
        });
        this.workspaceContainer.querySelectorAll('.connection-point').forEach(pointEl => {
            pointEl.classList.remove('highlighted');
        });
    },
    
    // Create connection between nodes
    createConnection: function(fromNodeId, toNodeId, condition, startPoint) {
        const fromNode = this.nodes.get(fromNodeId);
        const toNode = this.nodes.get(toNodeId);
        
        if (!fromNode || !toNode) return;
        
        // Check if connection with this condition already exists
        const existingConnection = fromNode.routes.find(route => 
            route.target === toNodeId && route.condition === condition
        );
        if (existingConnection) {
            Utils.showToast('Connection already exists', 'warning');
            return;
        }
        
        // Use the provided condition (from the output point)
        const actualCondition = condition === 'NEXT' ? null : condition;
        
        const route = {
            condition: actualCondition,
            target: toNodeId
        };
        
        fromNode.routes.push(route);
        
        // Find the actual start point if not provided
        const actualStartPoint = startPoint || this.findOutputPointByCondition(fromNodeId, condition);
        
        this.connections.set(`${fromNodeId}-${toNodeId}-${condition}`, {
            from: fromNodeId,
            to: toNodeId,
            condition: actualCondition,
            startPoint: actualStartPoint
        });
        
        this.renderConnection(fromNodeId, toNodeId, actualCondition, actualStartPoint);
        this.generateDSL();
        Utils.showToast('Connection created', 'success');
    },
    

    
    // Render connection line
    renderConnection: function(fromNodeId, toNodeId, condition, startPoint) {
        const fromNode = this.nodes.get(fromNodeId);
        const toNode = this.nodes.get(toNodeId);
        
        if (!fromNode || !toNode) return;
        
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        line.setAttribute('class', 'connection-line');
        line.setAttribute('marker-end', 'url(#arrowhead)');
        line.dataset.connectionId = `${fromNodeId}-${toNodeId}-${condition || 'NEXT'}`;
        
        this.updateConnectionPath(line, fromNode, toNode, startPoint);
        
        // Add click handler for connection editing (delete only now)
        line.addEventListener('click', (e) => {
            e.stopPropagation();
            this.deleteConnectionByLine(fromNodeId, toNodeId, condition || 'NEXT');
        });
        
        this.svg.appendChild(line);
        
        // No connection labels needed anymore since output points show the condition
    },
    
    // Update connection path
    updateConnectionPath: function(line, fromNode, toNode, startPoint) {
        let fromX = fromNode.x + 80; // Default: right edge of from node
        let fromY = fromNode.y + this.getNodeHeight(fromNode.type) / 2; // Default: center of from node
        
        // Get the connection condition from the line's dataset
        const connectionId = line.dataset.connectionId;
        const condition = connectionId ? connectionId.split('-')[2] : null;
        
                    // If we have a start point, calculate the exact position
            if (startPoint) {
                const nodeElement = this.workspaceContainer.querySelector(`[data-node-id="${fromNode.id}"]`);
                if (nodeElement) {
                    const rect = nodeElement.getBoundingClientRect();
                    const workspaceRect = this.workspaceContainer.getBoundingClientRect();
                    const pointRect = startPoint.getBoundingClientRect();
                    
                    fromX = fromNode.x + (pointRect.left + pointRect.width/2 - rect.left);
                    fromY = fromNode.y + (pointRect.top + pointRect.height/2 - rect.top);
                }
            } else {
                // Find the correct output point based on the condition
                const outputPoint = this.findOutputPointByCondition(fromNode.id, condition);
                if (outputPoint) {
                    const nodeElement = this.workspaceContainer.querySelector(`[data-node-id="${fromNode.id}"]`);
                    if (nodeElement) {
                        const nodeRect = nodeElement.getBoundingClientRect();
                        const workspaceRect = this.workspaceContainer.getBoundingClientRect();
                        const pointRect = outputPoint.getBoundingClientRect();
                        
                        // Calculate relative position within the node
                        const relativeX = pointRect.left + pointRect.width/2 - nodeRect.left;
                        const relativeY = pointRect.top + pointRect.height/2 - nodeRect.top;
                        
                        fromX = fromNode.x + relativeX;
                        fromY = fromNode.y + relativeY;
                    }
                }
            }
        
        const toX = toNode.x; // Left edge of to node
        const toY = toNode.y + this.getNodeHeight(toNode.type) / 2; // Center of to node
        
        // Handle self-connections with a loop
        if (fromNode.id === toNode.id) {
            // Create a self-loop that's clearly visible
            const nodeWidth = 80;
            const nodeHeight = this.getNodeHeight(fromNode.type);
            const loopSize = 150;
            
            // Start from the actual output point position
            const startX = fromX;
            const startY = fromY;
            
            // End at the input point (left side, center)
            const endX = fromNode.x;
            const endY = fromNode.y + nodeHeight / 2;
            
            // Create a pronounced loop
            const loopCenterX = startX + loopSize;
            const loopCenterY = fromNode.y + nodeHeight / 2;
            
            const controlX1 = loopCenterX;
            const controlY1 = startY - 150;
            const controlX2 = loopCenterX;
            const controlY2 = endY + 150;
            
            const path = `M ${startX} ${startY} C ${controlX1} ${controlY1}, ${controlX2} ${controlY2}, ${endX} ${endY}`;
            line.setAttribute('d', path);
        } else {
            // Create curved path for regular connections
            const controlX1 = fromX + (toX - fromX) * 0.5;
            const controlY1 = fromY;
            const controlX2 = fromX + (toX - fromX) * 0.5;
            const controlY2 = toY;
            
            const path = `M ${fromX} ${fromY} C ${controlX1} ${controlY1}, ${controlX2} ${controlY2}, ${toX} ${toY}`;
            line.setAttribute('d', path);
        }
    },
    

    
    // Delete connection by clicking on line
    deleteConnectionByLine: function(fromNodeId, toNodeId, condition) {
        if (confirm('Delete this connection?')) {
            this.deleteConnection(fromNodeId, toNodeId, condition);
        }
    },
    
    // Update node positions in DOM
    updateNodePositions: function() {
        this.nodes.forEach(node => {
            const nodeElement = this.workspaceContainer.querySelector(`[data-node-id="${node.id}"]`);
            if (nodeElement) {
                nodeElement.style.left = node.x + 'px';
                nodeElement.style.top = node.y + 'px';
            }
        });
    },
    
    // Update all connections (called when nodes move)
    updateConnections: function() {
        this.connections.forEach((connection, connectionId) => {
            const fromNode = this.nodes.get(connection.from);
            const toNode = this.nodes.get(connection.to);
            const line = this.svg.querySelector(`[data-connection-id="${connectionId}"]`);
            
            if (line && fromNode && toNode) {
                this.updateConnectionPath(line, fromNode, toNode, connection.startPoint);
            }
        });
    },
    

    
    // Delete connection
    deleteConnection: function(fromNodeId, toNodeId, condition = null) {
        const connectionId = `${fromNodeId}-${toNodeId}-${condition || 'NEXT'}`;
        this.connections.delete(connectionId);
        
        // Remove from node routes
        const fromNode = this.nodes.get(fromNodeId);
        if (fromNode) {
            const actualCondition = condition === 'NEXT' ? null : condition;
            fromNode.routes = fromNode.routes.filter(route => 
                !(route.target === toNodeId && route.condition === actualCondition)
            );
        }
        
        // Remove visual elements
        const line = this.svg.querySelector(`[data-connection-id="${connectionId}"]`);
        
        if (line) line.remove();
        
        this.generateDSL();
        Utils.showToast('Connection deleted', 'success');
    },
    
    // Update node ID
    updateNodeId: function(oldId, newId) {
        if (oldId === newId || !newId.trim()) return;
        
        // Check if new ID already exists
        if (this.nodes.has(newId)) {
            Utils.showToast('Node ID already exists', 'error');
            // Revert the input
            const input = this.workspaceContainer.querySelector(`[data-node-id="${oldId}"] .node-id`);
            if (input) input.value = oldId;
            return;
        }
        
        // Validate node ID format
        if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(newId)) {
            Utils.showToast('Invalid node ID format', 'error');
            const input = this.workspaceContainer.querySelector(`[data-node-id="${oldId}"] .node-id`);
            if (input) input.value = oldId;
            return;
        }
        
        const node = this.nodes.get(oldId);
        if (!node) return;
        
        // Update node
        node.id = newId;
        this.nodes.delete(oldId);
        this.nodes.set(newId, node);
        
        // Update DOM
        const nodeElement = this.workspaceContainer.querySelector(`[data-node-id="${oldId}"]`);
        if (nodeElement) {
            nodeElement.dataset.nodeId = newId;
        }
        
        // Update connections
        this.connections.forEach((connection, connectionId) => {
            if (connection.from === oldId) {
                connection.from = newId;
                const conditionPart = connection.condition || 'NEXT';
                const newConnectionId = `${newId}-${connection.to}-${conditionPart}`;
                this.connections.delete(connectionId);
                this.connections.set(newConnectionId, connection);
                
                // Update visual elements
                const line = this.svg.querySelector(`[data-connection-id="${connectionId}"]`);
                if (line) line.dataset.connectionId = newConnectionId;
            }
            
            if (connection.to === oldId) {
                connection.to = newId;
                const conditionPart = connection.condition || 'NEXT';
                const newConnectionId = `${connection.from}-${newId}-${conditionPart}`;
                this.connections.delete(connectionId);
                this.connections.set(newConnectionId, connection);
                
                // Update visual elements
                const line = this.svg.querySelector(`[data-connection-id="${connectionId}"]`);
                if (line) line.dataset.connectionId = newConnectionId;
            }
        });
        
        // Update all node routes that reference this node
        this.nodes.forEach(n => {
            n.routes.forEach(route => {
                if (route.target === oldId) {
                    route.target = newId;
                }
            });
        });
        
        this.generateDSL();
    },
    
    // Delete node
    deleteNode: function(nodeId) {
        const node = this.nodes.get(nodeId);
        if (!node) return;
        
        // Remove connections
        const connectionsToRemove = [];
        this.connections.forEach((connection, connectionId) => {
            if (connection.from === nodeId || connection.to === nodeId) {
                connectionsToRemove.push(connectionId);
            }
        });
        
        connectionsToRemove.forEach(connectionId => {
            const connection = this.connections.get(connectionId);
            const conditionKey = connection.condition || 'NEXT';
            this.deleteConnection(connection.from, connection.to, conditionKey);
        });
        
        // Remove from other nodes' routes
        this.nodes.forEach(n => {
            n.routes = n.routes.filter(route => route.target !== nodeId);
        });
        
        // Remove node
        this.nodes.delete(nodeId);
        
        // Remove visual element
        const nodeElement = this.workspaceContainer.querySelector(`[data-node-id="${nodeId}"]`);
        if (nodeElement) {
            nodeElement.remove();
        }
        
        this.generateDSL();
        Utils.showToast('Node deleted', 'success');
    },
    
    // Clear workspace
    clearWorkspace: function() {
        if (this.nodes.size === 0) return;
        
        if (confirm('Are you sure you want to clear the workspace?')) {
            this.nodes.clear();
            this.connections.clear();
            this.nodeCounter = 0;
            
            // Clear visual elements
            this.workspaceContainer.querySelectorAll('.factory-node').forEach(el => el.remove());
            this.svg.querySelectorAll('.connection-line').forEach(el => el.remove());
            
            this.generateDSL();
            Utils.showToast('Workspace cleared', 'success');
        }
    },
    
    // Center view
    centerView: function() {
        if (this.nodes.size === 0) return;
        
        // Calculate bounds of all nodes
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        
        this.nodes.forEach(node => {
            const nodeHeight = this.getNodeHeight(node.type);
            minX = Math.min(minX, node.x);
            minY = Math.min(minY, node.y);
            maxX = Math.max(maxX, node.x + 80);
            maxY = Math.max(maxY, node.y + nodeHeight);
        });
        
        // Calculate center offset
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const workspaceCenter = { x: this.workspace.offsetWidth / 2, y: this.workspace.offsetHeight / 2 };
        
        const offsetX = workspaceCenter.x - centerX;
        const offsetY = workspaceCenter.y - centerY;
        
        // Move all nodes
        this.nodes.forEach(node => {
            node.x += offsetX;
            node.y += offsetY;
            
            const nodeElement = this.workspaceContainer.querySelector(`[data-node-id="${node.id}"]`);
            if (nodeElement) {
                nodeElement.style.left = node.x + 'px';
                nodeElement.style.top = node.y + 'px';
            }
        });
        
        this.updateConnections();
    },
    
    // Zoom in function
    zoomIn: function() {
        if (this.zoomLevel >= this.maxZoom) {
            Utils.showToast('Maximum zoom level reached', 'info');
            return;
        }
        
        this.zoomLevel = Math.min(this.maxZoom, this.zoomLevel + this.zoomStep);
        this.applyZoom();
        Utils.showToast(`Zoomed in to ${Math.round(this.zoomLevel * 100)}%`, 'success');
    },
    
    // Zoom out function
    zoomOut: function() {
        if (this.zoomLevel <= this.minZoom) {
            Utils.showToast('Minimum zoom level reached', 'info');
            return;
        }
        
        this.zoomLevel = Math.max(this.minZoom, this.zoomLevel - this.zoomStep);
        this.applyZoom();
        Utils.showToast(`Zoomed out to ${Math.round(this.zoomLevel * 100)}%`, 'success');
    },
    
    // Apply zoom transform to workspace
    applyZoom: function() {
        // Scale the workspace container content
        this.workspaceContainer.style.transform = `scale(${this.zoomLevel})`;
        this.workspaceContainer.style.transformOrigin = '0 0';
        
        // Adjust the workspace dimensions to accommodate the scaled content
        const baseWidth = 4000; // Base workspace width
        const baseHeight = 3000; // Base workspace height
        const scaledWidth = baseWidth * this.zoomLevel;
        const scaledHeight = baseHeight * this.zoomLevel;
        
        this.workspaceContainer.style.width = baseWidth + 'px';
        this.workspaceContainer.style.height = baseHeight + 'px';
        this.workspace.style.width = scaledWidth + 'px';
        this.workspace.style.height = scaledHeight + 'px';
        
        // Update SVG dimensions
        this.svg.style.width = baseWidth + 'px';
        this.svg.style.height = baseHeight + 'px';
        
        // Update zoom button states
        const zoomInBtn = this.container.querySelector('.zoom-in-btn');
        const zoomOutBtn = this.container.querySelector('.zoom-out-btn');
        
        if (zoomInBtn) {
            zoomInBtn.disabled = this.zoomLevel >= this.maxZoom;
            zoomInBtn.title = this.zoomLevel >= this.maxZoom ? 
                'Maximum zoom reached' : `Zoom In (${Math.round((this.zoomLevel + this.zoomStep) * 100)}%)`;
        }
        
        if (zoomOutBtn) {
            zoomOutBtn.disabled = this.zoomLevel <= this.minZoom;
            zoomOutBtn.title = this.zoomLevel <= this.minZoom ? 
                'Minimum zoom reached' : `Zoom Out (${Math.round((this.zoomLevel - this.zoomStep) * 100)}%)`;
        }
    },
    
    // Reset zoom to 100%
    resetZoom: function() {
        this.zoomLevel = 1;
        this.applyZoom();
        Utils.showToast('Zoom reset to 100%', 'success');
    },
    
    // Generate DSL code from visual graph
    generateDSL: function() {
        if (this.nodes.size === 0) return '';
        
        let dsl = '';
        const processedNodes = new Set();
        
        // Start with START node
        const startNode = Array.from(this.nodes.values()).find(n => n.type === 'START');
        if (startNode) {
            dsl += this.generateNodeDSL(startNode);
            processedNodes.add(startNode.id);
        }
        
        // Process other nodes in order
        const sortedNodes = Array.from(this.nodes.values())
            .filter(n => n.type !== 'START')
            .sort((a, b) => {
                // END nodes last
                if (a.type === 'END' && b.type !== 'END') return 1;
                if (b.type === 'END' && a.type !== 'END') return -1;
                return a.id.localeCompare(b.id);
            });
        
        sortedNodes.forEach(node => {
            if (!processedNodes.has(node.id)) {
                dsl += '\n' + this.generateNodeDSL(node);
                processedNodes.add(node.id);
            }
        });
        
        return dsl;
    },
    
    // Generate DSL for a single node
    generateNodeDSL: function(node) {
        let dsl = `${node.type} ${node.id}`;
        
        if (node.type !== 'END') {
            dsl += ':';
            
            node.routes.forEach(route => {
                dsl += '\n    ';
                if (route.condition) {
                    dsl += `[${route.condition}] ${route.target}`;
                } else {
                    dsl += `NEXT ${route.target}`;
                }
            });
        }
        
        return dsl;
    },
    
    // Load DSL into visual builder
    loadDSL: function(dslCode) {
        if (!dslCode.trim()) return;
        
        try {
            this.clearWorkspace();
            
            const lines = dslCode.trim().split('\n')
                .map(line => line.trim())
                .filter(line => line && !line.startsWith('#'));
            
            let currentNode = null;
            let nodeCounter = 0;
            
            lines.forEach(line => {
                // Node declaration
                const nodeMatch = line.match(/^(START|END|PULLER_RB|PULLER_YG|PAINTER_RED|PAINTER_BLUE|PAINTER_YELLOW|PAINTER_GREEN)\s+(\w+):?/);
                if (nodeMatch) {
                    const [, nodeType, nodeId] = nodeMatch;
                    
                    // Create node with temporary position (will be repositioned later)
                    const node = {
                        id: nodeId,
                        type: nodeType,
                        x: 0,
                        y: 0,
                        routes: []
                    };
                    
                    this.nodes.set(nodeId, node);
                    currentNode = node;
                    nodeCounter++;
                    return;
                }
                
                // Route declaration
                const routeMatch = line.match(/^\[([RBYG]|EMPTY)\]\s+(\w+)$|^NEXT\s+(\w+)$/);
                if (routeMatch && currentNode) {
                    const condition = routeMatch[1] || null;
                    const target = routeMatch[2] || routeMatch[3];
                    
                    currentNode.routes.push({ condition, target });
                }
            });
            
            // Apply intelligent layout after parsing
            this.applyIntelligentLayout();
            
            // Render all nodes after positioning
            this.nodes.forEach(node => {
                this.renderNode(node);
            });
            
            // Create visual connections after all nodes are created
            setTimeout(() => {
                this.nodes.forEach(node => {
                    node.routes.forEach(route => {
                        if (this.nodes.has(route.target)) {
                            const conditionKey = route.condition || 'NEXT';
                            this.connections.set(`${node.id}-${route.target}-${conditionKey}`, {
                                from: node.id,
                                to: route.target,
                                condition: route.condition,
                                startPoint: null // Will be calculated during rendering
                            });
                            this.renderConnection(node.id, route.target, route.condition, null);
                        }
                    });
                });
                
                // Force update all connections to use correct output points
                setTimeout(() => {
                    this.updateConnections();
                }, 50);
            }, 100);
            
            Utils.showToast('DSL loaded successfully', 'success');
            
        } catch (error) {
            Utils.showToast('Error loading DSL: ' + error.message, 'error');
        }
    },
    
    // Handle keyboard events
    handleKeyDown: function(e) {
        if (e.key === 'Delete' && this.selectedNode) {
            this.deleteNode(this.selectedNode);
            this.selectedNode = null;
        } else if (e.key === 'Escape') {
            this.connectionMode = false;
            this.connectionStart = null;
            this.connectionStartCondition = null;
            this.connectionStartPoint = null;
            this.clearConnectionHighlights();
        } else if ((e.ctrlKey || e.metaKey) && (e.key === '=' || e.key === '+')) {
            e.preventDefault();
            this.zoomIn();
        } else if ((e.ctrlKey || e.metaKey) && e.key === '-') {
            e.preventDefault();
            this.zoomOut();
        } else if ((e.ctrlKey || e.metaKey) && e.key === '0') {
            e.preventDefault();
            this.resetZoom();
        }
    },
    
    // Handle workspace click
    handleWorkspaceClick: function(e) {
        if (e.target === this.workspace || e.target === this.workspaceContainer) {
            // Deselect all nodes
            this.workspaceContainer.querySelectorAll('.factory-node').forEach(node => {
                node.classList.remove('selected');
            });
            this.selectedNode = null;
            this.closeRouteConfig();
        }
    },
    
    // Handle workspace mouse events for selection
    handleWorkspaceMouseDown: function(e) {
        if (e.target.classList.contains('factory-node') || e.target.closest('.factory-node')) {
            const nodeElement = e.target.classList.contains('factory-node') ? 
                               e.target : e.target.closest('.factory-node');
            const nodeId = nodeElement.dataset.nodeId;
            
            // Select node (connection handling is now done through input/output points)
            this.workspaceContainer.querySelectorAll('.factory-node').forEach(node => {
                node.classList.remove('selected');
            });
            nodeElement.classList.add('selected');
            this.selectedNode = nodeId;
        }
    },
    
    handleWorkspaceMouseMove: function(e) {
        // Handled by individual node drag handlers
    },
    
    handleWorkspaceMouseUp: function(e) {
        // Handled by individual node drag handlers
    }
};

// Export for external use
window.ManufactoriaManager = {
    Utils,
    API,
    DSLEditor,
    FormValidator,
    SearchFilter,
    AutoSave,
    VisualFactoryBuilder
}; 