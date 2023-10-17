
  // Function to show the <div> after a delay
  function showDivWithDelay() {
    // Set the delay time in milliseconds (e.g., 2000ms for a 2-second delay)
    var delay = 2000;

    // Use setTimeout to display the <div> after the specified delay
    setTimeout(function () {
      document.getElementById("myDiv").style.display = "block";
    }, delay);
  }

  // Call the function to show the <div> with a delay when the page loads
  window.onload = showDivWithDelay;

