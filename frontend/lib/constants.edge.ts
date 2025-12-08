// lib/constants.edge.ts  (AMAN untuk Edge: tidak ada import Node)
export const isProductionEnvironment =
  process.env.NODE_ENV === "production";

export const isDevelopmentEnvironment =
  process.env.NODE_ENV === "development";

export const isTestEnvironment = Boolean(
  process.env.PLAYWRIGHT_TEST_BASE_URL ||
    process.env.PLAYWRIGHT ||
    process.env.CI_PLAYWRIGHT
);

// Sesuaikan dengan token email guest: "guest-123@..." → match
export const guestRegex = /^guest-\d+$/;
export const guestEmailRegex = /^guest-\d+@/i;
