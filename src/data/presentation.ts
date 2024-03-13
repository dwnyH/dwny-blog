type Social = {
  label: string;
  link: string;
};

type Presentation = {
  mail: string;
  title: string;
  description: string;
  socials: Social[];
  profile?: string;
};

const presentation: Presentation = {
  mail: "dawoonh316@gmail.com",
  title: "Hi, I’m Dawoon 👋",
  // profile: "/profile.webp",
  description:
    "저는 *React Native*로 앱 개발하고 있는 *프론트엔드 개발자*입니다. 공유 문화를 통해 IT기술이 빠르게 발전하고, 그런 기술들로 세상의 문제들이 하나씩 해결되는 과정에 흥미를 느끼고 있습니다. 🙂",
  socials: [
    {
      label: "Github",
      link: "https://github.com/dwnyH",
    },
  ],
};

export default presentation;
