interface LoadingSpinnerProps {
  color?: 'blue' | 'green' | 'purple' | 'yellow';
  size?: 'sm' | 'md' | 'lg';
  text?: string;
}

export default function LoadingSpinner({ 
  color = "blue", 
  size = "md", 
  text = "Loading..." 
}: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: "h-8 w-8",
    md: "h-12 w-12", 
    lg: "h-16 w-16"
  };

  const colorClasses = {
    blue: "border-blue-500",
    green: "border-green-500",
    purple: "border-purple-500",
    yellow: "border-yellow-500"
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-64 space-y-4">
      <div className="relative">
        <div className={`${sizeClasses[size]} ${colorClasses[color]} border-2 border-t-transparent rounded-full animate-spin`}></div>
        <div className={`absolute inset-0 ${sizeClasses[size]} ${colorClasses[color]} border-2 border-t-transparent rounded-full animate-ping opacity-20`}></div>
      </div>
      {text && (
        <div className="text-gray-300 text-sm font-medium animate-pulse">
          {text}
        </div>
      )}
    </div>
  );
} 