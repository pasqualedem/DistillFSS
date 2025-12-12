document.addEventListener('DOMContentLoaded', () => {
    
    const domains = [
        { 
            styleClass: 'is-medical',
            supportText: 'Medical Support',
            supportIcon: 'fas fa-heartbeat',
            studentName: 'Medical Student',
            studentIcon: 'fas fa-user-md' 
        },
        { 
            styleClass: 'is-industrial',
            supportText: 'Industrial Support',
            supportIcon: 'fas fa-industry',
            studentName: 'Industrial Student',
            studentIcon: 'fas fa-robot' 
        },
        { 
            styleClass: 'is-remote',
            supportText: 'Satellite Support',
            supportIcon: 'fas fa-satellite',
            studentName: 'Remote Sensing Student',
            studentIcon: 'fas fa-globe-americas' 
        }
    ];
    
    // Select Elements
    const animBox = document.getElementById('domain-anim-box');
    
    const supportCard = document.getElementById('anim-support');
    const supportLabel = document.getElementById('support-label');
    const supportIconWrapper = document.getElementById('support-icon-wrapper'); // Target the wrapper
    
    const teacherCard = document.getElementById('anim-teacher');
    const core = document.getElementById('anim-core');
    
    const studentCard = document.getElementById('anim-student');
    const studentLabel = document.getElementById('student-label');
    const studentIconWrapper = document.getElementById('student-icon-wrapper'); // Target the wrapper
    
    let currentDomainIndex = 0;

    function runAnimation() {
        // 1. Reset
        animBox.classList.remove('is-medical', 'is-industrial', 'is-remote');
        supportCard.classList.remove('animate-in');
        teacherCard.classList.remove('animate-in');
        core.classList.remove('pulse');
        studentCard.classList.remove('animate-out');

        void animBox.offsetWidth; // Force Reflow

        // 2. Load Content
        const domain = domains[currentDomainIndex];
        
        animBox.classList.add(domain.styleClass);
        
        // Update Text
        supportLabel.textContent = domain.supportText;
        studentLabel.textContent = domain.studentName;
        
        // Update Icons by replacing HTML (Forces FontAwesome to re-render)
        supportIconWrapper.innerHTML = `<i class="${domain.supportIcon}"></i>`;
        studentIconWrapper.innerHTML = `<i class="${domain.studentIcon}"></i>`;

        // 3. Trigger Animation
        supportCard.classList.add('animate-in');
        teacherCard.classList.add('animate-in');
        core.classList.add('pulse');
        studentCard.classList.add('animate-out');

        // 4. Next Index
        currentDomainIndex = (currentDomainIndex + 1) % domains.length;
    }

    runAnimation();
    setInterval(runAnimation, 6000); 
});