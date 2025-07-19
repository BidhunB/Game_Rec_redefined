import { User } from "@/contexts/AuthContext";

/**
 * Get a consistent user ID for API calls
 * Priority: user.id > user.email > fallback
 */
export function getUserId(user: User | null, fallback: string = "anonymous"): string {
  if (!user) {
    return fallback;
  }
  
  // Use user.id if available (from NextAuth)
  if (user.id && user.id !== user.email) {
    return user.id;
  }
  
  // Fall back to email if id is not available or same as email
  if (user.email) {
    return user.email;
  }
  
  return fallback;
}

/**
 * Get a user-friendly display name
 */
export function getUserDisplayName(user: User | null): string {
  if (!user) {
    return "Guest";
  }
  
  return user.name || user.email || "User";
}

/**
 * Check if user ID is valid (not anonymous or guest)
 */
export function isValidUserId(userId: string): boolean {
  return Boolean(userId && userId !== "anonymous" && userId !== "guest" && userId !== "user1");
}

/**
 * Sanitize user ID for database storage
 */
export function sanitizeUserId(userId: string): string {
  // Remove any special characters that might cause issues
  return userId.replace(/[^a-zA-Z0-9@._-]/g, '');
}

/**
 * Check if user is an admin
 * You can customize this logic based on your admin requirements
 */
export function isAdmin(user: User | null): boolean {
  if (!user || !user.email) {
    return false;
  }
  
  // Add your admin email addresses here
  const adminEmails = [
    "joeljoy1237@gmail.com", // Replace with your actual email
    // Add more admin emails as needed
  ];
  
  return adminEmails.includes(user.email.toLowerCase());
}

/**
 * Get admin status with additional info
 */
export function getAdminInfo(user: User | null): { isAdmin: boolean; reason?: string } {
  if (!user) {
    return { isAdmin: false, reason: "User not authenticated" };
  }
  
  if (!user.email) {
    return { isAdmin: false, reason: "No email available" };
  }
  
  const adminEmails = [
    "admin@example.com",
    "your-email@gmail.com", // Replace with your actual email
    // Add more admin emails as needed
  ];
  
  const isAdminUser = adminEmails.includes(user.email.toLowerCase());
  
  return {
    isAdmin: isAdminUser,
    reason: isAdminUser ? "Email in admin list" : "Email not in admin list"
  };
} 