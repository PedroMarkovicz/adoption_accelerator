import type { MetaResponse, PetProfileRequest } from "@/lib/types";

export interface ListingLabels {
  title: string;
  species: string;
  breed: string | null;
  colors: string | null;
  age: string;
  gender: string;
  size: string | null;
  fur: string | null;
  state: string | null;
  fee: string;
  health: { label: string; value: string }[];
}

function lookup(
  options: { id: number; name?: string; label?: string }[] | undefined,
  id: number,
): string | null {
  if (!options || !id) return null;
  const hit = options.find((o) => o.id === id);
  return hit ? (hit.name ?? hit.label ?? null) : null;
}

function formatAge(months: number): string {
  if (months < 12) return `${months} month${months === 1 ? "" : "s"}`;
  const years = Math.floor(months / 12);
  const rest = months % 12;
  const y = `${years} year${years === 1 ? "" : "s"}`;
  return rest === 0 ? y : `${y} ${rest} month${rest === 1 ? "" : "s"}`;
}

function ageBand(months: number): string {
  if (months < 12) return "young";
  if (months < 84) return "adult";
  return "senior";
}

export function buildListingLabels(
  profile: PetProfileRequest,
  meta: MetaResponse | undefined,
): ListingLabels {
  const species = profile.pet_type;
  const primary = lookup(meta?.breeds, profile.breed1);
  const secondary = lookup(meta?.breeds, profile.breed2 ?? 0);
  const breed = primary && secondary ? `${primary} / ${secondary}` : primary;

  const colors = [profile.color1, profile.color2, profile.color3]
    .map((id) => lookup(meta?.colors, id ?? 0))
    .filter((c): c is string => c !== null);

  const name = (profile.name ?? "").trim();
  const band = ageBand(profile.age_months);
  const lower = species.toLowerCase();
  const article = /^[aeiou]/i.test(band) ? "An" : "A";
  const title = name
    ? `Meet ${name}`
    : primary
      ? `${article} ${band} ${primary} ${lower}`
      : `${article} ${band} ${lower}`;

  return {
    title,
    species,
    breed,
    colors: colors.length > 0 ? colors.join(" / ") : null,
    age: formatAge(profile.age_months),
    gender: profile.gender,
    size: lookup(meta?.maturity_sizes, profile.maturity_size),
    fur: lookup(meta?.fur_lengths, profile.fur_length),
    state: lookup(meta?.states, profile.state ?? 0),
    fee: !profile.fee ? "Free" : `RM ${profile.fee}`,
    health: [
      { label: "Vaccinated", value: profile.vaccinated },
      { label: "Dewormed", value: profile.dewormed },
      { label: "Sterilized", value: profile.sterilized },
      { label: "Health", value: profile.health },
    ],
  };
}
