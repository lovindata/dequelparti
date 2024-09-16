import "./globals.css";
import { ThemeProvider } from "@/src/components/shadcn/theme-provider";
import { cn } from "@/src/lib/utils";
import type { Metadata } from "next";
import { Roboto } from "next/font/google";

const fontSans = Roboto({
  weight: ["100", "300", "400", "500", "700", "900"],
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "De quel parti ?",
  description: "C'est de quel parti politique ?",
  metadataBase: new URL("https://lovindata.github.io"), // For open graph resolution
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={cn(
          "min-h-screen bg-background antialiased",
          fontSans.className,
        )}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
