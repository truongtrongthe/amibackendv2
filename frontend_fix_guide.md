# Frontend Bug Fix Guide

## Issue 1: Social Media URL Rendering Error

### Error
```
ContactDetail.tsx:189 Uncaught TypeError: url.startsWith is not a function
    at ContactDetail.tsx:189:27
    at Array.map (<anonymous>)
    at renderSocialLinks (ContactDetail.tsx:185:17)
    at ContactDetail (ContactDetail.tsx:471:51)
```

### Problem
The error occurs because `url.startsWith` is being called, but `url` is not a string. This happens when the `social_media_urls` array from the API contains items where the `url` property isn't a string.

### Fix

1. **Update the `renderSocialLinks` function to check for valid URLs:**

```tsx
const renderSocialLinks = (socialMediaUrls) => {
  // First check if we have valid social media URLs
  if (!socialMediaUrls || !Array.isArray(socialMediaUrls)) {
    return null;
  }
  
  return socialMediaUrls.map((item, index) => {
    // Check if item and url exist and url is a string
    if (!item || typeof item.url !== 'string') {
      return null; // Skip invalid items
    }
    
    const url = item.url;
    const platform = item.platform || 'website';
    
    // Get appropriate icon based on platform
    let icon = 'globe'; // default
    if (platform === 'facebook') icon = 'facebook';
    if (platform === 'linkedin') icon = 'linkedin';
    if (platform === 'twitter') icon = 'twitter';
    
    return (
      <a 
        href={url.startsWith('http') ? url : `https://${url}`} 
        target="_blank" 
        rel="noopener noreferrer"
        key={index}
        className="social-link"
      >
        <i className={`fa fa-${icon}`}></i>
        {platform}
      </a>
    );
  });
};
```

2. **Alternative fix - in your API call for profile details:**

```tsx
// After fetching profile details
if (profileData && profileData.profile) {
  // Ensure social_media_urls is an array of valid objects
  if (profileData.profile.social_media_urls) {
    profileData.profile.social_media_urls = 
      profileData.profile.social_media_urls
        .filter(item => item && typeof item.url === 'string');
  } else {
    profileData.profile.social_media_urls = [];
  }
  
  setProfile(profileData.profile);
}
```

## Issue 2: Conversation Retrieval Error

### Error
```
GET http://localhost:5001/contact-conversations?contact_id=52&limit=5&offset=0 500 (INTERNAL SERVER ERROR)
```

### Solution
We've fixed this issue on the backend by:

1. Adding better error handling
2. Ensuring proper JSON serialization of conversation data
3. Adding fallbacks for missing data structures

If you're still seeing 500 errors when fetching conversations, please check:

1. **Network tab in DevTools**: Look for the exact error response
2. **Server logs**: Check the backend logs for specific error messages
3. **Retry with a different contact_id**: Try fetching conversations for other contacts

## How to Update Your Code to Handle API Variability

When working with APIs that return data from external sources like Facebook, it's good practice to add defensive coding to handle variable data formats:

```tsx
// Example of robust data handling in React component
useEffect(() => {
  const fetchData = async () => {
    try {
      const contactData = await getContactDetails(contactId);
      setContact(contactData.contact || {});
      
      if (contactData.contact?.id) {
        try {
          const convosData = await getContactConversations(
            contactData.contact.id,
            5,  // limit
            0   // offset
          );
          
          // Safely handle conversations data
          setConversations(convosData.conversations || []);
        } catch (error) {
          console.error("Error fetching conversations:", error);
          setConversations([]);
          // Show a non-blocking error message to user
          setConversationError("Unable to load conversations");
        }
        
        try {
          const profileData = await getProfileDetails(contactData.contact.id);
          
          // Safely extract and normalize profile
          const profile = profileData.profile || {};
          
          // Handle social media URLs safely
          if (profile.social_media_urls && Array.isArray(profile.social_media_urls)) {
            profile.social_media_urls = profile.social_media_urls
              .filter(item => item && typeof item.url === 'string');
          } else {
            profile.social_media_urls = [];
          }
          
          setProfile(profile);
        } catch (error) {
          console.error("Error fetching profile:", error);
          setProfile({});
        }
      }
    } catch (error) {
      console.error("Error fetching contact:", error);
      // Handle error state
    }
  };
  
  fetchData();
}, [contactId]);
``` 