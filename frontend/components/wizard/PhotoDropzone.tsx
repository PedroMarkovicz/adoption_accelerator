"use client";
import { useRef } from "react";

export function PhotoDropzone({ images, setImages }: { images: File[]; setImages: (f: File[]) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  function onFiles(list: FileList | null) {
    if (!list) return;
    setImages([...images, ...Array.from(list)].slice(0, 8));
  }
  return (
    <div>
      <button type="button" onClick={() => inputRef.current?.click()}
        className="w-full rounded-xl border-2 border-dashed border-ink/20 bg-surface px-6 py-10 text-sm text-muted hover:border-teal">
        Click to add photos (up to 8)
      </button>
      <input ref={inputRef} type="file" accept="image/*" multiple hidden onChange={(e) => onFiles(e.target.files)} />
      {images.length > 0 && (
        <div className="mt-4 grid grid-cols-4 gap-3">
          {images.map((f, i) => (
            <div key={i} className="relative">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img alt={`Upload ${i + 1}`} src={URL.createObjectURL(f)} className="h-20 w-full rounded-lg object-cover" />
              <button type="button" aria-label={`Remove photo ${i + 1}`} onClick={() => setImages(images.filter((_, j) => j !== i))}
                className="absolute right-1 top-1 rounded-full bg-ink px-2 text-xs text-paper">x</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
