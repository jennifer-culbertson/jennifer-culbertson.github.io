/* Copyright (c) 2013, Layar B.V. All rights reserved. 
   Content in this file is licensed under the MIT license.  
   http://opensource.org/licenses/MIT 
*/


$(document).ready(function() {
  //Initially hide all the expandable content with class name "content" 
  $(".contentexpandcollapse").hide();
  
  //Toggle the component when parent class "heading" is clicked on 
  $(".headingexpandcollapse").click(function() {
    
    var header = $(this);
    
    //Toggle the arrow image based on whether the content <p> is expanded or not
    header.children('#arrow').toggleClass('plus_sign minus_sign');
    
    //Expand or collapse the content <p> with slide mode animation 
    header.next(".contentexpandcollapse").slideToggle(500);
  });
});
