// Video Modal System
$(document).ready(function () {
  // Create modal HTML
  const modalHTML = `
    <div id="videoModal" class="video-modal" style="display: none;">
      <div class="video-modal-content">
        <span class="video-modal-close">&times;</span>
        <video id="modalVideo" controls>
          Your browser does not support the video tag.
        </video>
      </div>
    </div>
  `;
  
  // Add modal to body
  $('body').append(modalHTML);
  
  // Add click handlers to video previews
  $('.preview-video').click(function(e) {
    e.preventDefault();
    const videoSrc = $(this).find('video').attr('src');
    const video = $('#modalVideo')[0];
    video.src = videoSrc;
    video.load();
    $('#videoModal').fadeIn(300);
    video.play();
  });
  
  // Close modal
  $('.video-modal-close').click(function() {
    const video = $('#modalVideo')[0];
    video.pause();
    video.currentTime = 0;
    $('#videoModal').fadeOut(300);
  });
  
  // Close modal when clicking outside
  $('#videoModal').click(function(e) {
    if (e.target === this) {
      const video = $('#modalVideo')[0];
      video.pause();
      video.currentTime = 0;
      $('#videoModal').fadeOut(300);
    }
  });
  
  // Close modal with Escape key
  $(document).keydown(function(e) {
    if (e.key === "Escape" && $('#videoModal').is(':visible')) {
      const video = $('#modalVideo')[0];
      video.pause();
      video.currentTime = 0;
      $('#videoModal').fadeOut(300);
    }
  });
}); 