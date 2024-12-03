"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Upload,
  Loader2,
  Microscope,
  ShieldCheck,
  Dna,
  AlertTriangle,
} from "lucide-react";
import axios from "axios";

export function Page() {
  const [image, setImage] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImage(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!image) {
      console.log("Please select an image to upload.");
      return;
    }

    const formData = new FormData();
    formData.append("image", image);

    setIsLoading(true);
    setResult("");

    try {
      const response = await axios.post(
        "https://7355-2a01-cb00-b6c-3300-996d-df09-bc81-538d.ngrok-free.app",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      console.log("Response:", response.data);
      setResult(response.data.prediction);
      console.log("Response:", response.data);
    } catch (error) {
      console.error("Error analyzing image:", error);
      setResult(
        "An error occurred while analyzing the image. Please try again later."
      );
    }

    // Replace this with actual API call to your model
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen flex flex-col bg-emerald-50">
      <header className="bg-emerald-700 text-white shadow-md">
        <div className="container mx-auto px-4 py-6 flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <Microscope className="h-8 w-8" />
            <span className="text-2xl font-bold">SkinScanAI</span>
          </div>
          <nav>
            <ul className="flex space-x-4">
              <li>
                <a
                  href="#about"
                  className="hover:text-emerald-200 transition-colors"
                >
                  About
                </a>
              </li>
              <li>
                <a
                  href="#scan"
                  className="hover:text-emerald-200 transition-colors"
                >
                  Scan
                </a>
              </li>
              <li>
                <a
                  href="#technology"
                  className="hover:text-emerald-200 transition-colors"
                >
                  Technology
                </a>
              </li>
            </ul>
          </nav>
        </div>
      </header>

      <main className="flex-grow">
        <section className="bg-gradient-to-r from-emerald-600 to-emerald-800 py-20 text-center text-white">
          <div className="container mx-auto px-4">
            <h1 className="text-4xl font-bold mb-4">Welcome to SkinScanAI</h1>
            <p className="text-xl mb-8">
              Advanced skin cancer detection at your fingertips
            </p>
            <Button size="lg" variant="secondary" asChild>
              <a href="#scan">Start Your Scan</a>
            </Button>
          </div>
        </section>

        <section id="scan" className="py-20">
          <div className="container mx-auto px-4">
            <Card className="w-full max-w-2xl mx-auto bg-white shadow-lg">
              <CardHeader className="bg-emerald-700 text-white">
                <CardTitle>Skin Cancer Detection Scan</CardTitle>
                <CardDescription className="text-emerald-100">
                  Upload an image and provide details for analysis
                </CardDescription>
              </CardHeader>
              <CardContent className="mt-6">
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="image">Upload Skin Image</Label>
                    <div className="flex items-center justify-center w-full">
                      <label
                        htmlFor="image"
                        className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer bg-slate-50 hover:bg-slate-100"
                      >
                        <div className="flex flex-col items-center justify-center pt-5 pb-6">
                          <Upload className="w-8 h-8 mb-4 text-emerald-600" />
                          <p className="mb-2 text-sm text-slate-500">
                            <span className="font-semibold">
                              Click to upload
                            </span>{" "}
                            or drag and drop
                          </p>
                          <p className="text-xs text-slate-500">
                            PNG, JPG or JPEG (MAX. 800x400px)
                          </p>
                        </div>
                        <Input
                          id="image"
                          type="file"
                          className="hidden"
                          onChange={handleImageChange}
                          accept="image/*"
                        />
                      </label>
                    </div>
                  </div>

                  <Button
                    type="submit"
                    className="w-full bg-emerald-600 hover:bg-emerald-700 text-white"
                    disabled={isLoading || !image}
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Processing
                      </>
                    ) : (
                      "Analyze Image"
                    )}
                  </Button>
                </form>
              </CardContent>
              {result && (
                <CardFooter className="bg-amber-50 border-t border-amber-200">
                  <div className="flex items-start space-x-2">
                    <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5" />
                    <p className="text-sm text-amber-800">{result}</p>
                  </div>
                </CardFooter>
              )}
            </Card>
          </div>
        </section>

        <section id="technology" className="py-20 bg-emerald-100">
          <div className="container mx-auto px-4">
            <h2 className="text-3xl font-bold mb-8 text-center text-slate-800">
              Our Technology
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <Card className="bg-white shadow-md hover:shadow-lg transition-shadow duration-300">
                <CardHeader>
                  <CardTitle className="flex items-center text-emerald-700">
                    <Microscope className="mr-2 h-6 w-6" />
                    Advanced Imaging
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-slate-600">
                    Our AI utilizes state-of-the-art image processing techniques
                    to analyze skin lesions with high precision.
                  </p>
                </CardContent>
              </Card>
              <Card className="bg-white shadow-md hover:shadow-lg transition-shadow duration-300">
                <CardHeader>
                  <CardTitle className="flex items-center text-emerald-700">
                    <Dna className="mr-2 h-6 w-6" />
                    Machine Learning
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-slate-600">
                    Powered by sophisticated machine learning algorithms trained
                    on vast datasets of skin cancer images.
                  </p>
                </CardContent>
              </Card>
              <Card className="bg-white shadow-md hover:shadow-lg transition-shadow duration-300">
                <CardHeader>
                  <CardTitle className="flex items-center text-emerald-700">
                    <ShieldCheck className="mr-2 h-6 w-6" />
                    Privacy First
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-slate-600">
                    Your data security is our priority. All scans are processed
                    with strict privacy and security measures.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>
      </main>

      <footer className="bg-emerald-900 text-white py-8">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-semibold mb-4 text-emerald-300">
                About SkinScanAI
              </h3>
              <p className="text-sm text-slate-300">
                Pioneering AI-powered skin cancer detection for early diagnosis
                and improved outcomes.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-4 text-emerald-300">
                Quick Links
              </h3>
              <ul className="space-y-2 text-sm">
                <li>
                  <a
                    href="#"
                    className="text-slate-300 hover:text-emerald-300 transition-colors"
                  >
                    Home
                  </a>
                </li>
                <li>
                  <a
                    href="#scan"
                    className="text-slate-300 hover:text-emerald-300 transition-colors"
                  >
                    Scan
                  </a>
                </li>
                <li>
                  <a
                    href="#technology"
                    className="text-slate-300 hover:text-emerald-300 transition-colors"
                  >
                    Our Technology
                  </a>
                </li>
              </ul>
            </div>
          </div>
          <div className="mt-8 text-center text-sm text-slate-400">
            <p>&copy; 2024 SkinScanAI. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
